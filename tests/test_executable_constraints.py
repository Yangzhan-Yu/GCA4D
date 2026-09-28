"""Unit tests for the executable geometry constraint layer.

Run with:  python -m pytest tests/test_executable_constraints.py -q
or without pytest:  python tests/test_executable_constraints.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.constraints.executor import (  # noqa: E402
    obb_extent,
    percentile_extent,
    relative_direction,
    raw_extent,
    surface_distance_m,
)
from workflow.constraints.pipeline import run_operation, save_constraint  # noqa: E402
from workflow.constraints.task_constraints import compile_task_constraint  # noqa: E402
from workflow.constraints.validator import (  # noqa: E402
    map_value_to_option,
    validate_operation,
    validate_result,
)


def _cloud(center, sigma=0.02, n=400, seed=0):
    rng = np.random.RandomState(seed)
    return rng.normal(0, sigma, (n, 3)) + np.asarray(center, dtype=float)


def _constraint(operation, unit='m', version=1, options=None, frame_required=True):
    return {
        'question_id': 1,
        'question_type': 'test',
        'question': 'q',
        'version': version,
        'entities': [],
        'reference_frame': {'frame_type': 'world', 'coordinate_frame_id': 'world'},
        'time_constraint': {'mode': 'full_video'},
        'operation': {
            'operation': operation,
            'unit': unit,
            'options': options or [],
            'required_inputs': [],
        },
        'evidence': {
            'required_evidence': [],
            'coordinate_frame_required': frame_required,
        },
        'metadata': {},
    }


# ---------------------------------------------------------------- geometry
def test_direction_labels():
    origin = _cloud((0, 0, 0), sigma=0.001)
    forward = _cloud((0, 0, 1), sigma=0.001)
    cases = {
        'front': _cloud((0, 0, 2)),
        'right': _cloud((2, 0, 0)),
        'left': _cloud((-2, 0, 0)),
        'back': _cloud((0, 0, -2)),
    }
    for expected, target in cases.items():
        label = relative_direction(origin, forward, target)['label']
        assert label == expected, f'{expected} != {label}'


def test_direction_degenerate():
    origin = _cloud((0, 0, 0), sigma=0.001)
    forward = _cloud((0, 0, 0.001), sigma=0.001)
    target = _cloud((1, 0, 0))
    out = relative_direction(origin, forward, target)
    assert 'degenerate_direction' in out['quality_flags']


def test_surface_distance_known():
    a = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    b = np.array([[0.5, 0, 0], [0.6, 0, 0]])
    out = surface_distance_m(a, b)
    assert abs(out['distance_m'] - 0.3) < 1e-6


def test_extents():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 0.5, 0], [0, 0, 0.25]])
    assert abs(raw_extent(pts)['extent'] - 1.0) < 1e-6
    assert abs(percentile_extent(pts, 0, 100)['extent'] - 1.0) < 1e-6
    assert obb_extent(pts)['extent'] is not None


# --------------------------------------------------------------- validation
def test_validate_operation_rejects_duplicate_binding():
    constraint = _constraint('surface_distance')
    report = validate_operation(constraint, {
        'entity_bindings': {'entity_a': 'sofa_00', 'entity_b': 'sofa_00'},
        'coordinate_frame_id': 'world',
        'metric_scale': {'scale_factor': 1.0},
    })
    assert report['valid'] is False
    assert any(e['error_type'] == 'duplicate_instance_binding' for e in report['errors'])


def test_validate_operation_rejects_missing_metric_scale():
    constraint = _constraint('surface_distance')
    report = validate_operation(constraint, {
        'entity_bindings': {'entity_a': 'sofa_00', 'entity_b': 'stove_00'},
        'coordinate_frame_id': 'world',
        'metric_scale': {},
    })
    assert report['valid'] is False
    assert any(e['error_type'] == 'missing_metric_scale' for e in report['errors'])


def test_validate_operation_rejects_stale_geometry_version():
    constraint = _constraint('surface_distance')
    report = validate_operation(constraint, {
        'entity_bindings': {'entity_a': 'sofa_00', 'entity_b': 'stove_00'},
        'coordinate_frame_id': 'world',
        'metric_scale': {'scale_factor': 1.0},
        'declared_geometry_version': 'geo_old',
        'geometry_version': 'geo_new',
    })
    assert any(e['error_type'] == 'stale_geometry_version' for e in report['errors'])


def test_validate_operation_rejects_unbound_role():
    constraint = _constraint('relative_direction', unit='option', frame_required=True)
    report = validate_operation(constraint, {
        'entity_bindings': {'origin': 'fridge_00', 'forward': 'washer_00'},
        'coordinate_frame_id': 'world',
    })
    assert report['valid'] is False
    assert any(
        e['error_type'] == 'entity_not_bound' and e.get('role') == 'target'
        for e in report['errors']
    )


def test_validate_result_blocks_degenerate_distance():
    constraint = _constraint('surface_distance')
    report = validate_result(constraint, {
        'value': 0.0,
        'unit': 'm',
        'quality_flags': ['degenerate_distance', 'duplicate_masks_suspected'],
    })
    assert report['valid'] is False
    types = {e['error_type'] for e in report['errors']}
    assert {'degenerate_distance', 'duplicate_masks_suspected'} <= types


def test_validate_result_maps_option():
    constraint = _constraint('relative_direction', unit='option', options=['left', 'right'])
    report = validate_result(constraint, {
        'value': 'right',
        'unit': 'option',
        'options': ['left', 'right'],
        'quality_flags': [],
    })
    assert report['valid'] is True
    assert report['mapped_answer'] == 'right'


def test_map_value_to_option_numeric():
    assert map_value_to_option(2.9, ['2.7 m', '3.0 m', '2.9 m']) == '2.9 m'


# ------------------------------------------------------------------ compile
def test_compile_task_constraint_direction():
    plan = {
        'question_id': 680,
        'question_type': 'object_rel_direction_easy',
        'question': (
            'Standing by the sofa and facing the stove, is the chair to the '
            'left or right?'
        ),
        'target_entities': ['chair'],
        'reference_entities': ['sofa', 'stove'],
        'reference_frame': 'object:sofa',
        'required_evidence': [],
        'options': ['left', 'right'],
    }
    constraint = compile_task_constraint(plan)
    data = constraint.to_dict()
    assert data['operation']['operation'] == 'relative_direction'
    assert data['operation']['unit'] == 'option'
    assert data['reference_frame']['origin_entity'] == 'sofa'
    assert data['reference_frame']['forward_entity'] == 'stove'
    assert data['reference_frame']['target_entity'] == 'chair'


# ---------------------------------------------------------------- pipeline
def test_pipeline_happy_path(tmp_path):
    constraint = _constraint('surface_distance')
    points = {
        'entity_a': np.array([[0.0, 0, 0], [0.2, 0, 0]]),
        'entity_b': np.array([[0.5, 0, 0], [0.6, 0, 0]]),
    }
    report = run_operation(
        constraint,
        {'points': points, 'bindings': {'entity_a': 'sofa_00', 'entity_b': 'stove_00'}},
        context={
            'entity_bindings': {'entity_a': 'sofa_00', 'entity_b': 'stove_00'},
            'points': points,
            'metric_scale': {'scale_factor': 1.0},
            'coordinate_frame_id': 'world',
        },
        scene_root=tmp_path,
        question_id=1,
        persist=True,
    )
    assert report['status'] == 'valid', report
    assert abs(report['result']['value'] - 0.3) < 1e-6
    saved = json.loads((tmp_path / 'questions' / '1' / 'operation_results.json').read_text())
    assert 'surface_distance_01' in saved
    assert saved['surface_distance_01']['verification_status'] == 'verified'


def test_pipeline_rejects_duplicate_binding(tmp_path):
    constraint = _constraint('surface_distance')
    points = {
        'entity_a': np.array([[0.0, 0, 0]]),
        'entity_b': np.array([[0.0, 0, 0]]),
    }
    report = run_operation(
        constraint,
        {'points': points, 'bindings': {'entity_a': 'sofa_00', 'entity_b': 'sofa_00'}},
        context={'metric_scale': {'scale_factor': 1.0}, 'coordinate_frame_id': 'world'},
        scene_root=tmp_path,
        question_id=1,
        persist=True,
    )
    assert report['status'] == 'rejected'
    assert report['stage'] == 'validate_operation'
    assert not (tmp_path / 'questions' / '1' / 'operation_results.json').exists()


def test_save_constraint(tmp_path):
    plan = {
        'question_id': 7,
        'question_type': 'object_counting',
        'question': 'How many chairs?',
        'target_entities': ['chair'],
        'reference_entities': [],
        'options': [],
    }
    constraint = compile_task_constraint(plan)
    path = save_constraint(constraint, tmp_path, 7)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data['operation']['operation'] == 'count_instances'
    assert data['evidence']['coordinate_frame_required'] is False


def _main():
    import tempfile
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]
    failures = 0
    for name, func in tests:
        try:
            if func.__code__.co_argcount == 1:
                with tempfile.TemporaryDirectory() as tmp:
                    func(Path(tmp))
            else:
                func()
            print(f'PASS {name}')
        except Exception:
            failures += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
