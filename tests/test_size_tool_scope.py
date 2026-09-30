"""estimate_object_size must measure the whole object, not one track.

Regression: passing track_id filtered the observations, and the tool then wrote
the scaled SUBSET over ``{object_id}_points_metric.npz`` - the object's shared
metric cloud, which the constraint's object_extent reads.  A 1575-point track
silently replaced a 23708-point aggregate, and the extent came out unstable
(raw 45.8 / percentile 42.2 / obb 55.5 on a 62 cm stove).
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = (
    Path(__file__).resolve().parents[1]
    / 'entrypoints' / 'estimate_vsibench_object_size.py'
)


def _tree():
    return ast.parse(SOURCE.read_text(encoding='utf-8'))


def test_the_track_filter_helper_is_gone():
    names = {n.name for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef)}
    assert 'load_points_for_track' not in names, (
        'the track-filtering helper was the vector of the bug; it must not '
        'come back without rethinking the write-back'
    )


def test_the_measurement_loads_the_aggregated_cloud():
    source = SOURCE.read_text(encoding='utf-8')
    assert "f'{obj.object_id}_points.npz'" in source, (
        'the tool should load the object aggregate, not a per-track subset'
    )


def test_the_metric_cloud_write_is_not_track_scoped():
    """Nothing may filter observations by track_id before the write."""
    source = SOURCE.read_text(encoding='utf-8')
    assert "get('track_id') ==" not in source, (
        'filtering observations by track_id would reintroduce the subset '
        'overwrite'
    )


def test_result_distinguishes_object_id_from_requested_track():
    source = SOURCE.read_text(encoding='utf-8')
    assert "'track_id': obj.object_id" in source
    assert "'track_id_requested': args.track_id" in source


def _main():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failures = 0
    for name, func in tests:
        try:
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
