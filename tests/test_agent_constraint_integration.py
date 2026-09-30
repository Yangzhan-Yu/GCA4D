"""End-to-end check of the constraint tools wired into the agent entrypoint.

Requires the project environment (torch / cv2 / GroundingDINO imports).  Run:

    source scripts/gca_env.sh
    python tests/test_agent_constraint_integration.py

It builds a synthetic scene (no video, no models) and exercises the real tool
handlers: get_task_constraint -> execute_operation -> validate_result plus the
final-answer gate.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entrypoints.run_vsibench_agent import make_done_validator, register_tools  # noqa: E402
from tools.apis.four_d_memory import (  # noqa: E402
    MemoryObject,
    Observation,
    open_vsibench_memory,
)
from workflow.agentic.tool_registry import ToolRegistry  # noqa: E402
from evals.vsibench import VSIBench  # noqa: E402
from workflow.constraints.pipeline import save_constraint  # noqa: E402
from workflow.constraints.task_constraints import compile_task_constraint  # noqa: E402


def _grid(offset=(0.0, 0.0, 0.0)):
    base = np.array(
        [[x, y, z] for x in (-0.05, 0.05) for y in (0.0, 0.5) for z in (0.0, 0.5)],
        dtype=np.float64,
    )
    return base + np.asarray(offset, dtype=np.float64)


def _build_scene(root: Path):
    scene_root = root / 'results' / 'arkitscenes' / 'scene1'
    data_root = root / 'data' / 'vsibench'
    scene_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    store = open_vsibench_memory(
        data_root=data_root,
        dataset='arkitscenes',
        scene_name='scene1',
        scene_root=scene_root,
    )
    geometry_dir = store.root_dir / 'geometry'
    geometry_dir.mkdir(parents=True, exist_ok=True)
    clouds = {
        'sofa_00': _grid((0.0, 0.0, 0.0)),
        'stove_00': _grid((1.0, 0.0, 0.0)),
        'chair_00': _grid((0.0, 0.0, 1.0)),
    }
    for object_id, points in clouds.items():
        np.savez_compressed(geometry_dir / f'{object_id}_points_metric.npz', points=points)
    store.upsert_object(MemoryObject(object_id='sofa_00', category='sofa'))
    store.upsert_object(MemoryObject(object_id='stove_00', category='stove'))
    store.upsert_object(MemoryObject(object_id='chair_00', category='chair'))
    # Two chairs, each seen in two frames, so counting has real tracks to find.
    # Regression coverage: counting used to return 0 because the 'category'
    # role was bound to an instance id and 'tracks' was read from the wrong
    # input dict.
    for chair_id in ('chair_00', 'chair_01'):
        store.upsert_object(MemoryObject(object_id=chair_id, category='chair'))
        for index, frame_id in enumerate(('000900', '001080')):
            store.add_observation(Observation(
                observation_id=f'{chair_id}_{frame_id}',
                object_id=chair_id,
                frame_id=frame_id,
                timestamp=float(index * 3),
                confidence=0.9,
                position_3d=[float(index) * 0.2, 0.0, 0.0],
                metadata={'track_id': chair_id, 'point_count': 8000},
            ))
    evidence_dir = scene_root / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / 'metric_scale.json').write_text(
        json.dumps({'scale_factor': 1.0, 'method': 'synthetic'}) + '\n',
        encoding='utf-8',
    )
    return store, scene_root, data_root


def _context(store, scene_root, data_root, question_id, constraint,
             question_type='object_abs_distance'):
    return {
        'repo_root': Path(__file__).resolve().parents[1],
        'data_root': data_root,
        'results_root': scene_root.parents[1],
        'dataset': 'arkitscenes',
        'scene_name': 'scene1',
        'question_id': question_id,
        'question_type': question_type,
        'scene_root': scene_root,
        'store': store,
        'device': 'cpu',
        'task_constraint': constraint,
        'model': 'dummy',
    }


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store, scene_root, data_root = _build_scene(root)
        registry = register_tools(ToolRegistry())
        names = {spec.name for spec in registry.list_specs()}
        for required in (
            'get_task_constraint',
            'bind_constraint_entities',
            'validate_operation',
            'execute_operation',
            'validate_result',
        ):
            if required not in names:
                failures.append(f'missing tool {required}')

        plan = {
            'question_id': 680,
            'question_type': 'object_abs_distance',
            'question': 'What is the distance between the sofa and the stove?',
            'target_entities': ['sofa', 'stove'],
            'reference_entities': [],
            'options': [],
        }
        constraint = compile_task_constraint(plan).to_dict()
        save_constraint(constraint, scene_root, 680)
        context = _context(store, scene_root, data_root, 680, constraint)

        def call(name, **args):
            return registry.get(name).handler(context=context, **args)

        report = call('get_task_constraint')
        if report.get('operation') != 'surface_distance':
            failures.append(f'operation mismatch: {report.get("operation")}')
        if not report.get('validation', {}).get('valid'):
            failures.append(f'pre-validation failed: {report.get("validation")}')

        executed = call('execute_operation')
        if executed.get('status') != 'valid':
            failures.append(f'execute_operation status={executed.get("status")}: {executed}')
        value = (executed.get('result') or {}).get('value')
        if value is None or abs(value - 0.9) > 1e-3:
            failures.append(f'expected 0.9 m, got {value}')
        op_id = executed.get('operation_id')
        if not op_id:
            failures.append('execute_operation did not return an operation_id')

        verified = call('validate_result', operation_result_id=op_id)
        if verified.get('verification_status') != 'verified':
            failures.append(f'validate_result failed: {verified}')

        validator = make_done_validator(context)
        # With exactly one verified result the answer is taken from it, even if
        # the Planner omits the id or states a different number.
        auto = validator({'done': True, 'final_answer': 'nonsense'})
        if not auto.get('accepted'):
            failures.append(f'single verified result should auto-finalize: {auto}')
        elif auto.get('final_answer') != '0.9':
            failures.append(f'final answer not taken from verified result: {auto}')
        unknown = validator({
            'done': True,
            'final_answer': '0.9',
            'operation_result_id': 'surface_distance_99',
        })
        if unknown.get('accepted'):
            failures.append('unknown operation_result_id should be rejected')
        accepted = validator({
            'done': True,
            'final_answer': 'nonsense',
            'operation_result_id': op_id,
        })
        if not accepted.get('accepted'):
            failures.append(f'verified done rejected: {accepted}')
        if accepted.get('final_answer') != '0.9':
            failures.append(f'final answer not taken from verified result: {accepted}')

        # duplicate binding must be rejected with a targeted error type
        dup = call('execute_operation', bindings={'entity_a': 'sofa_00', 'entity_b': 'sofa_00'})
        if dup.get('status') != 'rejected':
            failures.append(f'duplicate binding not rejected: {dup}')
        else:
            types = {
                error['error_type']
                for error in dup.get('operation_validation', {}).get('errors', [])
            }
            if 'duplicate_instance_binding' not in types:
                failures.append(f'missing duplicate_instance_binding error: {types}')

        # Binding to an id without geometry must fail validation up front and
        # list the usable instances.  Regression: the availability check
        # compared the bindings against themselves, so it could never fail and
        # this surfaced later as an opaque empty_pointcloud rejection.
        bad_bind_plan = {
            'question_id': 685,
            'question_type': 'object_abs_distance',
            'question': 'distance between the sofa and the stove?',
            'target_entities': ['sofa', 'stove'],
            'reference_entities': [],
            'options': [],
        }
        bad_bind_constraint = compile_task_constraint(bad_bind_plan).to_dict()
        save_constraint(bad_bind_constraint, scene_root, 685)
        bad_bind_context = _context(
            store, scene_root, data_root, 685, bad_bind_constraint,
        )
        bad_bind_report = registry.get('execute_operation').handler(
            context=bad_bind_context,
            bindings={'entity_a': 'sofa_07', 'entity_b': 'stove_07'},
        )
        if bad_bind_report.get('status') != 'rejected':
            failures.append(
                f'binding to a geometry-less instance should be rejected: '
                f'{bad_bind_report}'
            )
        else:
            pre_errors = bad_bind_report['operation_validation']['errors']
            pre_types = {error['error_type'] for error in pre_errors}
            if 'entity_not_bound' not in pre_types:
                failures.append(f'expected entity_not_bound, got {pre_types}')
            if not any(
                error.get('available_instances') for error in pre_errors
            ):
                failures.append(
                    'the error must list available_instances so the Planner can '
                    f'rebind: {pre_errors}'
                )

        # counting must find the two chair tracks and report 2, not 0
        count_plan = {
            'question_id': 683,
            'question_type': 'object_counting',
            'question': 'How many chair(s) are in this room?',
            'target_entities': ['chair'],
            'reference_entities': [],
            'options': [],
        }
        count_constraint = compile_task_constraint(count_plan).to_dict()
        save_constraint(count_constraint, scene_root, 683)
        count_context = _context(
            store, scene_root, data_root, 683, count_constraint,
            question_type='object_counting',
        )
        count_report = registry.get('execute_operation').handler(context=count_context)
        if count_report.get('status') != 'valid':
            failures.append(f'count_instances failed: {count_report}')
        else:
            counted = count_report['result']['value']
            if counted != 2:
                failures.append(
                    f'expected count 2, got {counted} '
                    f"(metrics={count_report['result']['metrics']})"
                )
            if count_report['result']['entity_bindings'].get('category') != 'chair':
                failures.append(
                    'category role must bind to the category name: '
                    f"{count_report['result']['entity_bindings']}"
                )
            count_op_id = count_report['operation_id']
            count_done = make_done_validator(count_context)({
                'done': True,
                'final_answer': '2',
                'operation_result_id': count_op_id,
            })
            if not count_done.get('accepted') or count_done.get('final_answer') != '2':
                failures.append(f'counting answer not accepted: {count_done}')

        # The exact binding the local model produced must be rejected before
        # execution instead of silently counting 0.
        bad_cat = registry.get('execute_operation').handler(
            context=count_context,
            bindings={'category': 'chair_00,chair_01'},
        )
        if bad_cat.get('status') != 'rejected':
            failures.append(
                f'comma-joined category binding should be rejected: {bad_cat}'
            )
        else:
            errs = bad_cat['operation_validation']['errors']
            if 'invalid_category_binding' not in {
                e['error_type'] for e in errs
            }:
                failures.append(f'expected invalid_category_binding: {errs}')

        # relative direction over three bound instances
        dir_plan = {
            'question_id': 681,
            'question_type': 'object_rel_direction_easy',
            'question': (
                'Standing by the sofa and facing the stove, is the chair to the '
                'front or back?'
            ),
            'target_entities': ['chair'],
            'reference_entities': ['sofa', 'stove'],
            'options': ['left', 'right', 'front', 'back'],
        }
        dir_constraint = compile_task_constraint(dir_plan).to_dict()
        save_constraint(dir_constraint, scene_root, 681)
        dir_context = _context(store, scene_root, data_root, 681, dir_constraint)
        dir_report = registry.get('execute_operation').handler(context=dir_context)
        if dir_report.get('status') != 'valid':
            failures.append(f'relative_direction failed: {dir_report}')
        else:
            # sofa at origin, heading +X (towards stove), chair at +Z:
            # under the GCA +Y-down convention that is a 90 degree left turn.
            label = dir_report['result']['value']
            if label != 'left':
                failures.append(f'expected left, got {label}')

        # When only front/back are offered, a lateral target folds into the
        # nearer sector instead of being rejected - the question must still be
        # answerable with one of its own options.
        fold_plan = dict(dir_plan)
        fold_plan['question_id'] = 682
        fold_plan['options'] = ['front', 'back']
        fold_constraint = compile_task_constraint(fold_plan).to_dict()
        save_constraint(fold_constraint, scene_root, 682)
        fold_context = _context(store, scene_root, data_root, 682, fold_constraint)
        fold_report = registry.get('execute_operation').handler(context=fold_context)
        if fold_report.get('status') != 'valid':
            failures.append(f'front/back folding should be answerable: {fold_report}')
        elif fold_report['result']['value'] not in ('front', 'back'):
            failures.append(
                f"folded label must be one of the options, got "
                f"{fold_report['result']['value']!r}"
            )

        # An option set with no direction word at all cannot be mapped and
        # must be rejected rather than guessed.
        bad_plan = dict(dir_plan)
        bad_plan['question_id'] = 684
        bad_plan['options'] = ['A. yes', 'B. no']
        bad_constraint = compile_task_constraint(bad_plan).to_dict()
        save_constraint(bad_constraint, scene_root, 684)
        bad_context = _context(store, scene_root, data_root, 684, bad_constraint)
        bad_report = registry.get('execute_operation').handler(context=bad_context)
        if bad_report.get('status') != 'rejected':
            failures.append(f'unmappable answer should be rejected: {bad_report}')
        elif 'answer_not_mappable' not in {
            error['error_type']
            for error in bad_report.get('result_validation', {}).get('errors', [])
        }:
            failures.append(f'expected answer_not_mappable: {bad_report}')


        # The mapped multiple-choice answer must survive the official scorer,
        # which splits on the first space and strips the trailing dot.
        scorer = VSIBench._extract_mca_answer
        for option, expected_letter in (
            ('A. left', 'a'),
            ('B. right', 'b'),
            ('C. left', 'c'),
        ):
            got = scorer(option)
            if got != expected_letter:
                failures.append(
                    f'official scorer maps {option!r} -> {got!r}, expected '
                    f'{expected_letter!r}'
                )

    if failures:
        print('FAIL')
        for item in failures:
            print(f'  - {item}')
        return 1
    print('PASS agent constraint integration')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
