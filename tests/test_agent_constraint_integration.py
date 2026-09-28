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
from tools.apis.four_d_memory import MemoryObject, open_vsibench_memory  # noqa: E402
from workflow.agentic.tool_registry import ToolRegistry  # noqa: E402
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
    evidence_dir = scene_root / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / 'metric_scale.json').write_text(
        json.dumps({'scale_factor': 1.0, 'method': 'synthetic'}) + '\n',
        encoding='utf-8',
    )
    return store, scene_root, data_root


def _context(store, scene_root, data_root, question_id, constraint):
    return {
        'repo_root': Path(__file__).resolve().parents[1],
        'data_root': data_root,
        'results_root': scene_root.parents[1],
        'dataset': 'arkitscenes',
        'scene_name': 'scene1',
        'question_id': question_id,
        'question_type': 'object_abs_distance',
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

        # An answer that is not in the option set must be rejected.
        bad_plan = dict(dir_plan)
        bad_plan['question_id'] = 682
        bad_plan['options'] = ['front', 'back']
        bad_constraint = compile_task_constraint(bad_plan).to_dict()
        save_constraint(bad_constraint, scene_root, 682)
        bad_context = _context(store, scene_root, data_root, 682, bad_constraint)
        bad_report = registry.get('execute_operation').handler(context=bad_context)
        if bad_report.get('status') != 'rejected':
            failures.append(f'out-of-option answer should be rejected: {bad_report}')
        elif 'answer_not_mappable' not in {
            error['error_type']
            for error in bad_report.get('result_validation', {}).get('errors', [])
        }:
            failures.append(f'expected answer_not_mappable: {bad_report}')

    if failures:
        print('FAIL')
        for item in failures:
            print(f'  - {item}')
        return 1
    print('PASS agent constraint integration')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
