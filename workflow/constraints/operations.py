from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class OperationSpec:
    name: str
    unit: str
    required_inputs: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    description: str = ''
    # False = declared in the operation vocabulary but not executable yet.
    # Such questions fall back to open-ended tool use instead of failing with
    # an opaque "empty point cloud" error.
    implemented: bool = True


OPERATION_SPECS: Dict[str, OperationSpec] = {
    'relative_direction': OperationSpec(
        name='relative_direction',
        unit='option',
        required_inputs=['origin', 'forward', 'target', 'direction_boundaries'],
        required_evidence=['3d_points', 'coordinate_frame'],
        description='Horizontal direction of target relative to an origin/forward frame.',
    ),
    'surface_distance': OperationSpec(
        name='surface_distance',
        unit='m',
        required_inputs=['entity_a', 'entity_b'],
        required_evidence=['3d_points', 'metric_scale', 'coordinate_frame'],
        description='Closest surface distance between two bound object instances.',
    ),
    'argmin_distance': OperationSpec(
        name='argmin_distance',
        unit='option',
        required_inputs=['reference_entity', 'candidate_entities'],
        required_evidence=['3d_points', 'metric_scale', 'coordinate_frame'],
        description='Select the candidate entity with minimum surface distance.',
    ),
    'object_extent': OperationSpec(
        name='object_extent',
        unit='cm',
        required_inputs=['entity'],
        required_evidence=['3d_points', 'metric_scale', 'object_track'],
        description='Longest metric extent of one bound object track.',
    ),
    'count_instances': OperationSpec(
        name='count_instances',
        unit='count',
        required_inputs=['category'],
        required_evidence=['object_track', 'video_coverage'],
        description='Count unique physical instances of a category.',
    ),
    'first_visible_order': OperationSpec(
        name='first_visible_order',
        unit='option',
        required_inputs=['categories', 'time_intervals'],
        required_evidence=['event_timeline', 'video_coverage'],
        description='Order categories by first visible time interval.',
    ),
    'room_area': OperationSpec(
        name='room_area',
        unit='m2',
        required_inputs=['room_region'],
        required_evidence=['3d_points', 'metric_scale', 'room_region'],
        description='Metric area of the visible room/floor region.',
        implemented=False,
    ),
    'route_turns': OperationSpec(
        name='route_turns',
        unit='option',
        required_inputs=['start', 'heading', 'landmarks', 'route'],
        required_evidence=['scene_graph', 'coordinate_frame'],
        description='Turn sequence for navigation between bound landmarks.',
        implemented=False,
    ),
}


# Roles that bind to a category *name* (e.g. "chair") rather than to a
# concrete instance id (e.g. "chair_00").  Counting and timeline operations
# operate on every instance of a category, so there is no single instance to
# bind.
CATEGORY_ROLES = {'category', 'categories'}

# Leading option marker: "A. ", "B) ", "(C) ", "1. "
_OPTION_MARKER = re.compile(r'^\s*[\(\[]?[A-Za-z0-9][\)\].:]\s*')


def strip_option_marker(option: str) -> str:
    """Option text without its leading marker: ``'A. left'`` -> ``'left'``."""
    return _OPTION_MARKER.sub('', str(option)).strip()


# How good a mask has to be, per operation.  Missing entry = no bar.
#
#   locate  (count_instances, first_visible_order) - the object only has to be
#           found, so any mask works.
#   measure (object_extent, surface_distance, argmin_distance, room_area) - the
#           mask is lifted to 3D and measured.  A mask clipped by the image
#           border truncates the extent, and a fragmented one (spill,
#           reflection, occlusion) gives a partial point cloud.
#   orient  (relative_direction) - only the centroid matters, so the bar sits
#           between the two: a partly clipped object still has a usable centre.
MASK_QUALITY_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # An extent is the max dimension of the point cloud, so a clipped view
    # truncates it.  Reject only badly clipped views: measured on a real scene,
    # requiring "fully inside the frame" rejected 12 of 13 sofa views and left
    # no point cloud at all.
    'object_extent': {
        'max_mask_border_sides': 2,
        'min_mask_bbox_coverage': 0.50,
    },
    # The closest-surface distance only needs the surfaces facing the other
    # object.  Large furniture seen up close is routinely clipped, so a
    # truncation limit here rejects everything and the question becomes
    # unanswerable.  Fragmentation still disqualifies a mask.
    'surface_distance': {
        'max_mask_border_sides': 4,
        'min_mask_bbox_coverage': 0.40,
    },
    'argmin_distance': {
        'max_mask_border_sides': 4,
        'min_mask_bbox_coverage': 0.40,
    },
    'room_area': {
        'max_mask_border_sides': 4,
        'min_mask_bbox_coverage': 0.30,
    },
    # Direction uses the centroid, so a clipped view still has a usable centre.
    'relative_direction': {
        'max_mask_border_sides': 2,
        'min_mask_bbox_coverage': 0.35,
    },
}


def infer_operation(question_type: str, relation_or_metric: str = '') -> Optional[str]:
    question_type = str(question_type).strip().lower()
    mapping = {
        'object_rel_direction_easy': 'relative_direction',
        'object_rel_direction_medium': 'relative_direction',
        'object_rel_direction_hard': 'relative_direction',
        'object_abs_distance': 'surface_distance',
        'object_rel_distance': 'argmin_distance',
        'object_size_estimation': 'object_extent',
        'object_counting': 'count_instances',
        'obj_appearance_order': 'first_visible_order',
        'room_size_estimation': 'room_area',
        'route_planning': 'route_turns',
    }
    if question_type in mapping:
        return mapping[question_type]
    text = str(relation_or_metric).lower()
    if 'distance' in text and 'closest' in text:
        return 'surface_distance'
    return None


__all__ = [
    'OperationSpec',
    'OPERATION_SPECS',
    'CATEGORY_ROLES',
    'MASK_QUALITY_REQUIREMENTS',
    'strip_option_marker',
    'infer_operation',
]
