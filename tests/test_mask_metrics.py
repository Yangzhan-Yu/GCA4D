"""Mask shape metrics and the task-level bar they feed.

Counting only needs the object located, so any mask works.  Size and distance
lift the mask to 3D and measure it, so a mask clipped by the image border
truncates the extent and a fragmented one yields a partial point cloud.  These
tests pin both the metrics and the per-task thresholds.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.utils.mask_metrics import mask_quality, mask_rejection_reason  # noqa: E402
from workflow.constraints.task_constraints import compile_task_constraint  # noqa: E402


def _rect(height=400, width=600, y0=100, y1=300, x0=200, x1=400):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_intact_mask_scores_well():
    stats = mask_quality(_rect())
    assert stats['mask_boundary_ratio'] == 0.0
    assert stats['mask_bbox_coverage'] > 0.95


def test_bbox_border_sides_counts_clipping():
    """Scale independent: how many bbox sides sit on the image edge.

    The area-based boundary_ratio cannot express this.  A 400x200 rectangle
    flushed against the right edge only puts 2% of its pixels in the border
    band, so a 0.35 threshold never fires no matter how the object is clipped.
    Measured on a real scene: a sofa close to the camera was clipped on 1-3
    sides in 12 of 13 views, a small appliance on 0-1.
    """
    one_side = mask_quality(_rect(x1=600))
    assert one_side['mask_bbox_border_sides'] == 1
    assert one_side['mask_boundary_ratio'] < 0.1     # weak, as documented

    corner = mask_quality(_rect(x1=600, y0=0))
    assert corner['mask_bbox_border_sides'] == 2

    inside = mask_quality(_rect(x1=590, y1=100))
    assert inside['mask_bbox_border_sides'] == 0


def test_fragmented_mask_has_low_bbox_coverage():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    mask[80, 80] = True                           # two stray pixels
    stats = mask_quality(mask)
    assert stats['mask_bbox_coverage'] < 0.01


# --------------------------------------------------------------- the bar
def test_counting_sets_no_mask_bar():
    constraint = compile_task_constraint({
        'question_id': 1, 'question_type': 'object_counting',
        'question': 'how many chairs', 'target_entities': ['chair'],
        'reference_entities': [], 'options': [],
    }).to_dict()
    evidence = constraint['evidence']
    assert evidence['min_mask_bbox_coverage'] is None
    assert evidence['max_mask_border_sides'] is None


def _evidence_for(question_type, entities):
    return compile_task_constraint({
        'question_id': 1, 'question_type': question_type,
        'question': 'q', 'target_entities': entities,
        'reference_entities': [], 'options': [],
    }).to_dict()['evidence']


def test_size_tolerates_one_clipped_side_but_not_more():
    """An extent is truncated by clipping, but requiring "fully inside"
    rejected 12 of 13 real sofa views - so the limit is a few sides, not zero.
    """
    evidence = _evidence_for('object_size_estimation', ['stove'])
    assert evidence['max_mask_border_sides'] == 2
    assert evidence['min_mask_bbox_coverage'] == 0.50


def test_distance_sets_no_truncation_limit():
    """Closest-surface only needs the facing surfaces, and large furniture
    seen up close is routinely clipped.  A truncation limit here made the
    question unanswerable (no point cloud at all).
    """
    for question_type in ('object_abs_distance', 'object_rel_distance'):
        evidence = _evidence_for(question_type, ['sofa', 'stove'])
        assert evidence['max_mask_border_sides'] == 4, question_type


def test_direction_sits_between_the_two():
    constraint = compile_task_constraint({
        'question_id': 1, 'question_type': 'object_rel_direction_easy',
        'question': 'q', 'target_entities': ['tv'],
        'reference_entities': ['sofa', 'stove'], 'options': ['A. left', 'B. right'],
    }).to_dict()
    evidence = constraint['evidence']
    assert evidence['max_mask_border_sides'] == 2
    assert evidence['min_mask_bbox_coverage'] == 0.35


def test_rejection_reason_none_when_no_bar():
    assert mask_rejection_reason(mask_quality(_rect(x1=600))) is None


def test_rejection_reason_respects_the_per_operation_side_limit():
    one_side = mask_quality(_rect(x1=600))
    corner = mask_quality(_rect(x1=600, y0=0))

    # size: one clipped side is tolerable, a clipped corner is not
    assert mask_rejection_reason(one_side, max_mask_border_sides=2) is None
    assert mask_rejection_reason(corner, max_mask_border_sides=2) is None

    # distance sets no truncation limit: closest-surface only needs the facing
    # surfaces, and large furniture is routinely clipped
    assert mask_rejection_reason(one_side, max_mask_border_sides=4) is None
    assert mask_rejection_reason(corner, max_mask_border_sides=4) is None


def test_a_heavily_clipped_mask_is_rejected_for_size():
    stats = mask_quality(_rect(x0=0, x1=600, y0=0, y1=300))   # 3 sides
    assert stats['mask_bbox_border_sides'] == 3
    reason = mask_rejection_reason(stats, max_mask_border_sides=2)
    assert reason is not None and 'clipped by the frame' in reason


def test_rejection_reason_flags_a_fragmented_mask():
    """Two stray pixels far apart: huge bbox, almost nothing filled."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    mask[80, 80] = True
    stats = mask_quality(mask)
    assert stats['mask_bbox_coverage'] < 0.01
    reason = mask_rejection_reason(stats, min_bbox_coverage=0.5)
    assert reason is not None and 'fragmented' in reason


def test_rejection_reason_passes_an_intact_mask():
    stats = mask_quality(_rect())
    assert mask_rejection_reason(stats, 0.35, 0.5) is None


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
