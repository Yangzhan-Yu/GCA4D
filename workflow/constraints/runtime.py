"""Runtime glue between the constraint layer and the 4D scene memory.

The Planner binds semantic roles (``origin``, ``entity_a``, ``category`` ...)
to concrete instance ids.  This module turns those bindings into resolved point
clouds / track lists that ``execute_operation`` can consume, and persists the
bindings so a run is reproducible.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflow.constraints.operations import CATEGORY_ROLES
from workflow.constraints.pipeline import load_pointcloud, question_dir

# Roles that hold point-cloud evidence.
POINT_ROLES: Dict[str, List[str]] = {
    'surface_distance': ['entity_a', 'entity_b'],
    'relative_direction': ['origin', 'forward', 'target'],
    'object_extent': ['entity'],
    'argmin_distance': ['reference_entity'],
}

METRIC_OPERATIONS = {
    'surface_distance',
    'argmin_distance',
    'object_extent',
    'relative_direction',
    'room_area',
}


# A track counts as a distinct physical instance when it was seen in more
# than one frame, or when a single-frame detection was confident.
#
# The old rule also required >= 5000 points for a single-frame track.  That is
# a property of how large the object looks, not of whether it is real, and it
# silently dropped a 0.945-confidence chair that occupied a small part of the
# frame - turning a correct count of 2 into 1.  Detection confidence is the
# signal that actually separates a real single-frame object from a false
# positive.
MIN_TRACK_SUPPORT = 2
MIN_SINGLE_FRAME_CONFIDENCE = 0.70


def is_reliable_track(
    track: Dict[str, Any],
    min_support: int = MIN_TRACK_SUPPORT,
    min_single_frame_confidence: float = MIN_SINGLE_FRAME_CONFIDENCE,
) -> bool:
    support = len(track.get('frame_ids') or [])
    if support >= min_support:
        return True
    confidence = float(track.get('mean_confidence') or 0.0)
    return confidence >= min_single_frame_confidence


def reliable_tracks(tracks, **kwargs) -> List[Dict[str, Any]]:
    return [track for track in tracks if is_reliable_track(track, **kwargs)]


def default_bindings(constraint: Dict[str, Any]) -> Dict[str, str]:
    """One candidate binding per role, derived from the constraint's entities.

    Instance roles bind to ``<category>_00``; category roles (``category``,
    ``categories``) bind to the category name itself, because those operations
    span every instance of the category.  Binding a category role to an
    instance id would silently match zero tracks.
    """
    bindings: Dict[str, str] = {}
    for entity in constraint.get('entities', []):
        role = str(entity.get('role', '')).strip()
        category = str(entity.get('category', '')).strip().lower()
        if not role or not category:
            continue
        if role in CATEGORY_ROLES:
            bindings.setdefault(role, category)
        else:
            bindings.setdefault(role, f'{category}_00')
    return bindings


def bindings_path(scene_root, question_id) -> Path:
    return question_dir(scene_root, question_id) / 'entity_bindings.json'


def load_bindings(scene_root, question_id) -> Dict[str, Any]:
    path = bindings_path(scene_root, question_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def save_bindings(scene_root, question_id, bindings: Dict[str, Any]) -> Path:
    path = bindings_path(scene_root, question_id)
    path.write_text(
        json.dumps(bindings, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def merge_bindings(constraint, stored, incoming=None) -> Dict[str, Any]:
    merged = default_bindings(constraint)
    merged.update({k: v for k, v in (stored or {}).items() if v})
    merged.update({k: v for k, v in (incoming or {}).items() if v})
    return merged


def resolve_points(
    store_root,
    operation: str,
    bindings: Dict[str, Any],
    object_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load the point clouds needed by one operation.

    Returns a dict keyed by role.  Missing instances are returned as empty
    arrays so the validator can report ``entity_not_bound`` / ``empty_pointcloud``
    instead of raising.
    """
    store_root = Path(store_root)
    metric = operation in METRIC_OPERATIONS
    points: Dict[str, Any] = {}

    for role in POINT_ROLES.get(operation, []):
        instance_id = bindings.get(role)
        if not instance_id:
            continue
        cloud = load_pointcloud(store_root, str(instance_id), metric=metric)
        if cloud is None:
            continue
        points[role] = cloud

    if operation == 'argmin_distance':
        candidates = bindings.get('candidate_entities')
        if isinstance(candidates, str):
            candidates = [candidates]
        if not candidates and object_ids:
            reference = bindings.get('reference_entity')
            candidates = [item for item in object_ids if item != reference]
        resolved: Dict[str, Any] = {}
        for instance_id in candidates or []:
            cloud = load_pointcloud(store_root, str(instance_id), metric=metric)
            if cloud is not None:
                resolved[str(instance_id)] = cloud
        points['candidate_entities'] = resolved

    return points


def resolve_extra_inputs(
    operation: str,
    bindings: Dict[str, Any],
    args: Dict[str, Any],
    store=None,
) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if operation == 'count_instances':
        category = bindings.get('category')
        raw_tracks: List[Dict[str, Any]] = []
        if store is not None and category:
            for track in store.query_tracks(category=str(category)):
                raw_tracks.append({
                    'track_id': track.get('track_id'),
                    'support': len(track.get('frame_ids') or []),
                    'total_points': track.get('total_points'),
                    'mean_confidence': track.get('mean_confidence'),
                    'frame_ids': track.get('frame_ids') or [],
                    'centroid': track.get('centroid'),
                })
        # Reliability policy carried over from the former
        # count_entities_in_video tool: a track counts if it is seen in more
        # than one frame, or is dense enough and confident enough in a single
        # frame.  Both the filtered and unfiltered counts are reported so a
        # disagreement is visible instead of silent.
        reliable = reliable_tracks(raw_tracks)
        extra['tracks'] = reliable or raw_tracks
        extra['tracks_all'] = raw_tracks
    if operation == 'first_visible_order':
        intervals = {}
        if store is not None:
            for category in bindings.get('categories', []) or []:
                observations = store.query_observations()
                times = [
                    obs.timestamp for obs in observations
                    if getattr(obs, 'object_id', '').startswith(str(category))
                ]
                intervals[str(category)] = min(times) if times else None
        extra['time_intervals'] = intervals
    # 'method' is deliberately NOT taken from args: the caller overrides it
    # from the evidence profile so the measurement method cannot vary per call.
    if 'method' in args:
        print(
            f'[Profile] ignoring Planner-supplied method={args["method"]!r}',
            flush=True,
        )
    if 'vertical_axis' in args:
        extra['vertical_axis'] = args['vertical_axis']
    if 'direction_boundaries' in args:
        extra['direction_boundaries'] = args['direction_boundaries']
    return extra


__all__ = [
    'default_bindings',
    'load_bindings',
    'save_bindings',
    'merge_bindings',
    'resolve_points',
    'resolve_extra_inputs',
    'POINT_ROLES',
    'METRIC_OPERATIONS',
    'is_reliable_track',
    'reliable_tracks',
    'MIN_TRACK_SUPPORT',
    'MIN_SINGLE_FRAME_CONFIDENCE',
]
