"""Mask quality metrics, dependency-free (numpy only).

Lives outside ``tools.apis`` on purpose: the SAM3 worker runs in a separate
environment that has torch and numpy but none of the rest of the gca stack, and
it loads this by file path.
"""

from typing import Any, Dict

import numpy as np

# Minimum share of mask pixels sitting on the image border band for the mask to
# count as touching the frame edge (guards against stray pixels).
BORDER_CONTACT_MIN = 0.01


def mask_quality(mask) -> Dict[str, float]:
    """Shape statistics that predict how usable a mask is.

    * ``mask_area_ratio``   - fraction of the image covered.
    * ``mask_boundary_ratio`` - fraction of the mask that touches the image
      border.  High means the object is cut off, so any 3D extent measured from
      it is truncated.
    * ``mask_bbox_coverage`` - how much of the mask's bounding box is filled.
      Low means a fragmented mask (spill, reflection, occlusion).

    Counting only needs the object located, so all three can be poor.  Size and
    distance measure the object, so a truncated or fragmented mask is fatal -
    which is why the cache keeps several candidates and records these numbers
    instead of committing to the single best-scoring one.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    area = int(mask.sum())
    total = int(mask.size)
    area_ratio = area / max(1, total)

    boundary_width = max(1, int(min(height, width) * 0.02))
    boundary_region = np.zeros_like(mask, dtype=bool)
    boundary_region[:boundary_width, :] = True
    boundary_region[-boundary_width:, :] = True
    boundary_region[:, :boundary_width] = True
    boundary_region[:, -boundary_width:] = True
    boundary_pixels = int((mask & boundary_region).sum())
    boundary_ratio = boundary_pixels / max(1, area)

    rows, cols = np.where(mask)
    if area == 0:
        bbox_coverage = 0.0
    else:
        bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        bbox_coverage = area / max(1, bbox_area)

    # ``boundary_ratio`` is an *area* ratio, so a large object clipped by one
    # edge still scores ~0.02 and a 0.35 threshold never fires.  Counting the
    # bounding-box sides that sit on the image edge is scale independent and
    # actually answers "is this object running out of frame?" - measured on a
    # real scene, a sofa close to the camera had 1-3 sides clipped in 12 of 13
    # views while a small appliance had 0-1.
    if area == 0:
        bbox_border_sides = 0
    else:
        bbox_border_sides = (
            int(rows.min() <= 0)
            + int(rows.max() >= height - 1)
            + int(cols.min() <= 0)
            + int(cols.max() >= width - 1)
        )
    touches_border = boundary_ratio >= BORDER_CONTACT_MIN

    return {
        'mask_area_ratio': float(area_ratio),
        'mask_boundary_ratio': float(boundary_ratio),
        'mask_bbox_coverage': float(bbox_coverage),
        'mask_touches_border': bool(touches_border),
        'mask_bbox_border_sides': int(bbox_border_sides),
    }


def mask_rejection_reason(
    stats: Dict[str, float],
    max_boundary_ratio=None,
    min_bbox_coverage=None,
    max_mask_border_sides=None,
):
    """Why this mask is unfit for *measurement*, or None if it passes.

    The thresholds come from the task constraint.  Counting declares none - the
    object only has to be located.  Size and distance require a mask that is
    neither cut off by the frame edge (which truncates the measured extent) nor
    fragmented (which yields a partial point cloud).
    """
    if max_mask_border_sides is not None:
        sides = int(stats.get('mask_bbox_border_sides') or 0)
        if sides > int(max_mask_border_sides):
            return (
                f'mask bounding box is clipped by the frame on {sides} side(s) '
                f'(limit {max_mask_border_sides}); too much of the object is '
                'off-screen to measure it'
            )
    if max_boundary_ratio is not None:
        boundary = float(stats.get('mask_boundary_ratio') or 0.0)
        if boundary > float(max_boundary_ratio):
            return (
                f'mask touches the image border on {boundary:.2f} of its '
                f'perimeter (limit {max_boundary_ratio:.2f}); the object is '
                'cut off so any extent measured from it is truncated'
            )
    if min_bbox_coverage is not None:
        coverage = float(stats.get('mask_bbox_coverage') or 0.0)
        if coverage < float(min_bbox_coverage):
            return (
                f'mask fills only {coverage:.2f} of its bounding box (minimum '
                f'{min_bbox_coverage:.2f}); it is fragmented'
            )
    return None


__all__ = ['mask_quality', 'mask_rejection_reason', 'BORDER_CONTACT_MIN']
