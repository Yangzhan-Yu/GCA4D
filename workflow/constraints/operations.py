from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OperationSpec:
    name: str
    unit: str
    required_inputs: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    description: str = ''


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
    ),
    'route_turns': OperationSpec(
        name='route_turns',
        unit='option',
        required_inputs=['start', 'heading', 'landmarks', 'route'],
        required_evidence=['scene_graph', 'coordinate_frame'],
        description='Turn sequence for navigation between bound landmarks.',
    ),
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


__all__ = ['OperationSpec', 'OPERATION_SPECS', 'infer_operation']
