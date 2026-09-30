"""Which object tracks count as distinct physical instances.

A single-frame track used to need >= 5000 points.  That is a property of how
large the object looks, not of whether it is real, and it dropped a
0.945-confidence chair that occupied a small part of the frame - turning a
correct count of 2 into 1.  Detector confidence is the signal that separates a
real single-frame object from a false positive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.constraints.runtime import (  # noqa: E402
    MIN_SINGLE_FRAME_CONFIDENCE,
    is_reliable_track,
    reliable_tracks,
)


def _track(frames, points, confidence):
    return {
        'track_id': 'x',
        'frame_ids': list(frames),
        'total_points': points,
        'mean_confidence': confidence,
    }


def test_multi_frame_track_is_always_reliable():
    assert is_reliable_track(_track(['a', 'b'], points=10, confidence=0.1))


def test_confident_single_frame_track_is_reliable():
    """The chair that was dropped: one frame, few points, high confidence."""
    assert is_reliable_track(_track(['001350'], points=2769, confidence=0.945))


def test_low_confidence_single_frame_track_is_dropped():
    """The old spurious detection: 33k points but only 0.696 confidence."""
    assert not is_reliable_track(_track(['003240'], points=33466, confidence=0.696))


def test_the_microwave_detected_as_a_stove_is_dropped():
    assert not is_reliable_track(_track(['000840'], points=9000, confidence=0.435))


def test_point_count_no_longer_gates_a_confident_track():
    """Points describe apparent size, not realness."""
    sparse = _track(['a'], points=1, confidence=0.99)
    assert is_reliable_track(sparse)


def test_thresholds_are_tunable():
    track = _track(['a'], points=0, confidence=0.55)
    assert not is_reliable_track(track)
    assert is_reliable_track(track, min_single_frame_confidence=0.5)
    assert MIN_SINGLE_FRAME_CONFIDENCE == 0.70


def test_reliable_tracks_filters_a_mixed_list():
    tracks = [
        _track(['a', 'b'], 10, 0.2),
        _track(['c'], 100, 0.9),
        _track(['d'], 100, 0.3),
    ]
    kept = reliable_tracks(tracks)
    assert len(kept) == 2
    assert all(t['mean_confidence'] >= 0.7 or len(t['frame_ids']) >= 2 for t in kept)


def test_clustering_needs_extents_and_says_so():
    """Missing extents must not silently become one instance per observation.

    Regression: observations carried no extent, so every radius was 0, every
    threshold was 0, no pair merged, and counting returned one instance per
    frame - 7 chairs for a room holding 2, 11 tables for a room holding 4.
    """
    from tools.apis.four_d_memory import cluster_positions_by_extent

    positions = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.5, 0.0, 0.0]]
    try:
        cluster_positions_by_extent(positions, [None, None, None])
    except ValueError as exc:
        assert 'refusing to guess' in str(exc)
    else:
        raise AssertionError('missing extents should raise, not over-split')

    # an explicit fallback is allowed, and still merges the near pair
    clusters = cluster_positions_by_extent(
        positions, [None, None, None], fallback_eps=0.2
    )
    assert len(clusters) == 2, clusters


def test_near_views_of_one_object_merge():
    from tools.apis.four_d_memory import cluster_positions_by_extent

    # three views of one table, each seeing only part of it
    positions = [[0.278, 0.153, 0.601],
                 [0.372, 0.155, 0.587],
                 [0.409, 0.150, 0.621]]
    extents = [[0.127, 0.046, 0.157],
               [0.266, 0.060, 0.205],
               [0.268, 0.315, 0.281]]
    clusters = cluster_positions_by_extent(positions, extents)
    assert len(clusters) == 1, clusters


def test_distant_views_stay_separate():
    from tools.apis.four_d_memory import cluster_positions_by_extent

    positions = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]
    extents = [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]
    clusters = cluster_positions_by_extent(positions, extents)
    assert len(clusters) == 2, clusters


def test_the_collector_records_an_extent_per_observation():
    source = (Path(__file__).resolve().parents[1]
              / 'entrypoints' / 'collect_vsibench_evidence.py'
              ).read_text(encoding='utf-8')
    assert "'point_extent':" in source, (
        'observations must carry their own extent for track clustering'
    )


def test_link_scale_is_not_a_knife_edge():
    """The merge rule must be stable across a range of scales.

    Regression data (real observations): seven chair views of two chairs, and
    eleven table views of four tables.  Summing the two radii is the "bounding
    spheres just touch" test and chains everything together; taking the larger
    radius gives the correct count over a plateau rather than at a single
    setting.
    """
    from tools.apis.four_d_memory import cluster_positions_by_extent

    def count(positions, extents, scale):
        return len(cluster_positions_by_extent(positions, extents, scale=scale))

    # two objects, clearly apart
    positions = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.5, 0.0, 0.0]]
    extents = [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]
    for scale in (0.75, 1.0, 1.25):
        assert count(positions, extents, scale) == 2, scale

    # summing the radii would merge these two; taking the larger must not
    near = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    small = [[0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]
    assert count(near, small, 1.0) == 2, (
        'two 0.26 m objects 0.5 m apart are distinct'
    )


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
