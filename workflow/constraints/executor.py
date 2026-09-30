"""Deterministic execution of executable geometry constraints.

This module implements the ``execute_operation`` gap interface from
``GCA4D_VSI_Bench_Executable_Constraints_Plan.md``.  The Planner (Qwen) only
chooses *which* operation to run and *which* instances to bind to each role.
All numeric geometry is computed here with fixed, reviewable code so that every
reported value has an explicit coordinate frame, unit and evidence provenance.

The executor is intentionally free of heavy model imports: it operates on
already-reconstructed point clouds and can therefore be unit tested without a
GPU, VGGT or SAM.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from workflow.constraints.operations import OPERATION_SPECS, strip_option_marker


DEFAULT_DIRECTION_BOUNDARIES = {
    'front_deg': 45.0,
    'back_deg': 135.0,
}


@dataclass
class OperationResult:
    operation_id: str
    operation: str
    constraint_version: int
    unit: str
    value: Any
    entity_bindings: Dict[str, Any] = field(default_factory=dict)
    input_evidence_ids: List[str] = field(default_factory=list)
    coordinate_frame_id: Optional[str] = None
    geometry_version: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    stability: Dict[str, Any] = field(default_factory=dict)
    verification_status: str = 'unverified'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'operation': self.operation,
            'constraint_version': self.constraint_version,
            'unit': self.unit,
            'value': self.value,
            'entity_bindings': self.entity_bindings,
            'input_evidence_ids': self.input_evidence_ids,
            'coordinate_frame_id': self.coordinate_frame_id,
            'geometry_version': self.geometry_version,
            'quality_flags': self.quality_flags,
            'metrics': self.metrics,
            'options': self.options,
            'stability': self.stability,
            'verification_status': self.verification_status,
        }


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def to_points(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float64)
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected an (N, 3) point cloud, got shape {points.shape}')
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
    points = to_points(points)
    if len(points) == 0:
        raise ValueError('Cannot compute a centroid of an empty point cloud')
    return points.mean(axis=0)


def surface_distance_m(points_a, points_b) -> Dict[str, Any]:
    """Minimum surface (closest point) distance between two point clouds."""
    from scipy.spatial import cKDTree

    a = to_points(points_a)
    b = to_points(points_b)
    if len(a) == 0 or len(b) == 0:
        return {
            'distance_m': None,
            'point_a': None,
            'point_b': None,
            'index_a': None,
            'index_b': None,
            'quality_flags': ['empty_pointcloud'],
        }
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    dist_b, index_b = tree_b.query(a, k=1)
    best_a = int(np.argmin(dist_b))
    best_distance = float(dist_b[best_a])
    best_b = int(index_b[best_a])
    dist_a, index_a = tree_a.query(b, k=1)
    best_b_other = int(np.argmin(dist_a))
    if float(dist_a[best_b_other]) < best_distance:
        best_distance = float(dist_a[best_b_other])
        best_b = best_b_other
        best_a = int(index_a[best_b_other])

    centroid_distance = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    sample_a = a[:: max(1, len(a) // 3000)]
    sample_b = b[:: max(1, len(b) // 3000)]
    nearest_b, _ = tree_b.query(sample_a, k=1)
    nearest_a, _ = tree_a.query(sample_b, k=1)
    overlap = max(
        float(np.mean(nearest_b <= 0.05)),
        float(np.mean(nearest_a <= 0.05)),
    )
    return {
        'distance_m': best_distance,
        'point_a': a[best_a].tolist(),
        'point_b': b[best_b].tolist(),
        'index_a': best_a,
        'index_b': best_b,
        'centroid_distance_m': centroid_distance,
        'duplicate_mask_overlap': overlap,
        'quality_flags': [],
    }


def percentile_extent(points, lower: float = 5.0, upper: float = 95.0) -> Dict[str, Any]:
    pts = to_points(points)
    if len(pts) == 0:
        return {'extent': None, 'per_axis': None, 'quality_flags': ['empty_pointcloud']}
    low = np.percentile(pts, lower, axis=0)
    high = np.percentile(pts, upper, axis=0)
    per_axis = (high - low)
    return {
        'extent': float(np.max(per_axis)),
        'per_axis': per_axis.tolist(),
        'quality_flags': [],
    }


def raw_extent(points) -> Dict[str, Any]:
    pts = to_points(points)
    if len(pts) == 0:
        return {'extent': None, 'per_axis': None, 'quality_flags': ['empty_pointcloud']}
    per_axis = pts.max(axis=0) - pts.min(axis=0)
    return {
        'extent': float(np.max(per_axis)),
        'per_axis': per_axis.tolist(),
        'quality_flags': [],
    }


def obb_extent(points) -> Dict[str, Any]:
    pts = to_points(points)
    if len(pts) < 4:
        return {'extent': None, 'per_axis': None, 'quality_flags': ['empty_pointcloud']}
    centered = pts - pts.mean(axis=0)
    covariance = np.cov(centered.T)
    try:
        _, basis = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return {'extent': None, 'per_axis': None, 'quality_flags': ['degenerate_pointcloud']}
    projected = centered @ basis
    per_axis = projected.max(axis=0) - projected.min(axis=0)
    return {
        'extent': float(np.max(per_axis)),
        'per_axis': per_axis.tolist(),
        'quality_flags': [],
    }


def relative_direction(
    origin_points,
    forward_points,
    target_points,
    vertical_axis: Sequence[float] = (0.0, 1.0, 0.0),
    boundaries: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Horizontal direction of ``target`` seen from ``origin`` facing ``forward``.

    ``vertical_axis`` follows the GCA object-frame convention
    (``+Y`` points down).  With that axis, a *positive* signed angle means the
    target lies to the **right** of the heading.
    """
    bounds = dict(DEFAULT_DIRECTION_BOUNDARIES)
    bounds.update(boundaries or {})
    axes = np.asarray(vertical_axis, dtype=np.float64)
    norm = np.linalg.norm(axes)
    if norm == 0:
        raise ValueError('vertical_axis must be a non-zero vector')
    axes = axes / norm

    try:
        origin = centroid(origin_points)
        forward = centroid(forward_points)
        target = centroid(target_points)
    except ValueError as exc:
        return {'label': None, 'angle_deg': None, 'quality_flags': ['empty_pointcloud'], 'error': str(exc)}

    heading = forward - origin
    heading_h = heading - np.dot(heading, axes) * axes
    to_target = target - origin
    to_target_h = to_target - np.dot(to_target, axes) * axes

    flags: List[str] = []
    if np.linalg.norm(heading_h) < 0.05:
        flags.append('degenerate_direction')
    if np.linalg.norm(to_target_h) < 0.05:
        flags.append('degenerate_target')
    if flags:
        return {
            'label': None,
            'angle_deg': None,
            'quality_flags': flags,
            'heading_norm': float(np.linalg.norm(heading_h)),
            'target_norm': float(np.linalg.norm(to_target_h)),
        }

    numeric = float(np.dot(np.cross(heading_h, to_target_h), axes))
    signed = math.degrees(math.atan2(numeric, float(np.dot(heading_h, to_target_h))))
    if signed <= -180.0:
        signed += 360.0
    label = _direction_label(signed, bounds)
    return {
        'label': label,
        'angle_deg': signed,
        'quality_flags': [],
        'heading_norm': float(np.linalg.norm(heading_h)),
        'target_norm': float(np.linalg.norm(to_target_h)),
        'boundaries': bounds,
    }


DIRECTION_LABELS = ('front', 'back', 'left', 'right')

# Angular centre of each canonical direction label, in the GCA object-frame
# convention where a *positive* signed angle means "to the right".
_CANONICAL_DIRECTIONS = (
    (('front',), 0.0),
    (('front', 'right'), 45.0),
    (('right',), 90.0),
    (('back', 'right'), 135.0),
    (('back',), 180.0),
    (('back', 'left'), -135.0),
    (('left',), -90.0),
    (('front', 'left'), -45.0),
)


def _label_words(label: str):
    return set(re.split(r'[\s\-_]+', str(label).lower())) & set(DIRECTION_LABELS)


def _option_centre(label: str):
    """Angular centre of an answer option, or None if it names no direction."""
    words = _label_words(label)
    if not words:
        return None
    for canonical, angle in _CANONICAL_DIRECTIONS:
        if set(canonical) == words:
            return angle
    return None


def allowed_direction_labels(options) -> set:
    """The direction labels the answer options actually offer.

    Returns the option payloads (marker stripped), so a compound option such as
    ``"A. front-left"`` yields ``"front-left"``.
    """
    labels = set()
    for option in options or []:
        payload = strip_option_marker(option).lower().strip()
        if payload and _label_words(payload):
            labels.add(payload)
    return labels


def _direction_label(angle_deg: float, bounds: Dict[str, Any]) -> str:
    """Name the sector the target falls in, using the options as the sectors.

    Rather than hardcoding 45/135 degree cut-offs, each option is placed at its
    angular centre and the target is assigned to the nearest one.  That
    reproduces VSI-Bench's own boundaries - a back/right/left question splits
    back from right at 135 degrees, which is exactly what its wording says -
    and it also handles compound options such as "front-left".
    """
    labels = list(bounds.get('allowed') or [])
    centres = [
        (label, _option_centre(label)) for label in labels
    ]
    centres = [(label, angle) for label, angle in centres if angle is not None]
    if not centres:
        # No options supplied: report the full four-way labelling rather than
        # collapsing everything to left/right.
        centres = [
            ('front', 0.0), ('right', 90.0), ('back', 180.0), ('left', -90.0),
        ]

    def angular_distance(centre: float) -> float:
        return abs((angle_deg - centre + 180.0) % 360.0 - 180.0)

    return min(centres, key=lambda item: angular_distance(item[1]))[0]



# --------------------------------------------------------------------------
# result store
# --------------------------------------------------------------------------
class OperationResultStore:
    """Append-only store of executed operations for one question."""

    def __init__(self, path):
        self.path = Path(path)
        self._results: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._results = json.loads(self.path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                self._results = {}

    def next_operation_id(self, operation: str) -> str:
        index = 1
        while f'{operation}_{index:02d}' in self._results:
            index += 1
        return f'{operation}_{index:02d}'

    def add(self, result: Dict[str, Any]) -> str:
        operation_id = result['operation_id']
        self._results[operation_id] = result
        self.flush()
        return operation_id

    def update(self, operation_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._results.setdefault(operation_id, {})
        entry.update(patch)
        self.flush()
        return entry

    def get(self, operation_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(operation_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._results)

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._results, ensure_ascii=False, indent=2, default=_json_default) + '\n',
            encoding='utf-8',
        )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _pointcloud_digest(points) -> str:
    pts = to_points(points)
    if len(pts) == 0:
        return 'empty'
    summary = json.dumps(
        {
            'n': int(len(pts)),
            'mean': [round(float(v), 6) for v in pts.mean(axis=0)],
            'min': [round(float(v), 6) for v in pts.min(axis=0)],
            'max': [round(float(v), 6) for v in pts.max(axis=0)],
        },
        sort_keys=True,
    )
    return hashlib.sha1(summary.encode('utf-8')).hexdigest()[:12]



def _aux(inputs: Dict[str, Any], points_by_role: Dict[str, Any], key: str, default=None):
    """Read an auxiliary (non point-cloud) input.

    ``points_by_role`` only holds role -> point cloud.  Auxiliary payloads such
    as ``tracks`` or ``time_intervals`` are supplied at the top level of
    ``inputs``; falling back to ``points_by_role`` keeps older callers working.
    """
    if key in inputs:
        return inputs[key]
    return points_by_role.get(key, default)

# --------------------------------------------------------------------------
# operation dispatch
# --------------------------------------------------------------------------
def execute_operation(
    constraint: Any,
    inputs: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one operation against already-resolved evidence.

    ``inputs`` keys
    --------------
    points : dict
        role -> point cloud (N, 3).  For ``argmin_distance`` the
        ``candidate_entities`` entry is a dict of instance_id -> points.
    bindings : dict
        role -> instance id, copied into the result for provenance.
    options : list
        answer options for multiple-choice operations.
    """
    context = context or {}
    data = constraint.to_dict() if hasattr(constraint, 'to_dict') else dict(constraint or {})
    operation = (data.get('operation') or {}).get('operation')
    spec = OPERATION_SPECS.get(operation)
    if spec is None:
        raise ValueError(f'Unsupported operation: {operation!r}')

    points_by_role: Dict[str, Any] = dict(inputs.get('points') or {})
    bindings = dict(inputs.get('bindings') or {})
    options = list(inputs.get('options') or (data.get('operation') or {}).get('options') or [])
    unit = spec.unit
    quality_flags: List[str] = []
    metrics: Dict[str, Any] = {}
    stability: Dict[str, Any] = {}

    if operation == 'surface_distance':
        outcome = surface_distance_m(
            points_by_role.get('entity_a'),
            points_by_role.get('entity_b'),
        )
        value = outcome['distance_m']
        value = None if value is None else round(float(value), 4)
        quality_flags = list(outcome.get('quality_flags') or [])
        metrics = {
            'centroid_distance_m': outcome.get('centroid_distance_m'),
            'duplicate_mask_overlap': outcome.get('duplicate_mask_overlap'),
            'point_a': outcome.get('point_a'),
            'point_b': outcome.get('point_b'),
        }
        if value is not None and value < 0.05:
            quality_flags.append('degenerate_distance')
        if (
            metrics.get('duplicate_mask_overlap') is not None
            and metrics['duplicate_mask_overlap'] >= 0.5
        ):
            quality_flags.append('duplicate_masks_suspected')
        if (
            metrics.get('centroid_distance_m') is not None
            and metrics['centroid_distance_m'] < 0.10
        ):
            quality_flags.append('near_identical_pointclouds')

    elif operation == 'argmin_distance':
        reference = points_by_role.get('reference_entity')
        candidates = points_by_role.get('candidate_entities') or {}
        if not isinstance(candidates, dict):
            raise ValueError('argmin_distance requires candidate_entities as a dict')
        scored = []
        for instance_id, candidate_points in candidates.items():
            outcome = surface_distance_m(reference, candidate_points)
            scored.append({
                'instance_id': instance_id,
                'distance_m': outcome['distance_m'],
                'quality_flags': outcome.get('quality_flags') or [],
            })
        scored = [item for item in scored if item['distance_m'] is not None]
        scored.sort(key=lambda item: item['distance_m'])
        if not scored:
            value = None
            metrics = {'scored': []}
            quality_flags.append('empty_pointcloud')
        else:
            value = scored[0]['instance_id']
            metrics = {'scored': scored}
            distances = [item['distance_m'] for item in scored]
            if len(distances) > 1:
                spread = float(distances[0] - distances[1])
                stability = {'spread': spread, 'tolerance': 0.05}

    elif operation == 'object_extent':
        entity_points = points_by_role.get('entity')
        raw = raw_extent(entity_points)
        pct = percentile_extent(entity_points)
        obb = obb_extent(entity_points)
        method = str(inputs.get('method', 'percentile'))
        selected = {
            'raw': raw,
            'percentile': pct,
            'obb': obb,
        }.get(method, pct)
        extent_m = selected['extent']
        value = None if extent_m is None else round(float(extent_m) * 100.0, 2)
        unit = 'cm'
        metrics = {
            'raw_extent_cm': None if raw['extent'] is None else round(raw['extent'] * 100.0, 2),
            'percentile_extent_cm': None if pct['extent'] is None else round(pct['extent'] * 100.0, 2),
            'obb_extent_cm': None if obb['extent'] is None else round(obb['extent'] * 100.0, 2),
            'method': method,
            'point_count': int(len(to_points(entity_points))),
        }
        for key in ('raw_extent_cm', 'percentile_extent_cm', 'obb_extent_cm'):
            if metrics[key] is None:
                quality_flags.append('empty_pointcloud')
        extents = [
            item for item in (
                metrics['raw_extent_cm'],
                metrics['percentile_extent_cm'],
                metrics['obb_extent_cm'],
            ) if item is not None
        ]
        if len(extents) >= 2:
            spread = float(max(extents) - min(extents))
            tolerance = float(0.25 * np.median(extents))
            stability = {
                'spread': spread,
                'tolerance': tolerance,
            }
            # The validator blocks on this; record the flag too so the reason
            # is visible in the result without reading the validation report.
            if spread > tolerance:
                quality_flags.append('unstable_extent')

    elif operation == 'count_instances':
        tracks = _aux(inputs, points_by_role, 'tracks') or []
        all_tracks = _aux(inputs, points_by_role, 'tracks_all') or tracks
        value = int(len(tracks))
        unit = 'count'
        metrics = {
            'tracks': tracks,
            'track_count': len(tracks),
            'track_count_unfiltered': len(all_tracks),
        }
        # If a quality filter removed tracks, say so rather than silently
        # reporting a smaller number.
        if len(all_tracks) > len(tracks):
            quality_flags.append('tracks_filtered')

    elif operation == 'relative_direction':
        boundaries = dict(inputs.get('direction_boundaries') or {})
        boundaries.setdefault('allowed', sorted(allowed_direction_labels(options)))
        outcome = relative_direction(
            points_by_role.get('origin'),
            points_by_role.get('forward'),
            points_by_role.get('target'),
            vertical_axis=inputs.get('vertical_axis', (0.0, 1.0, 0.0)),
            boundaries=boundaries,
        )
        value = outcome.get('label')
        unit = 'option'
        quality_flags = list(outcome.get('quality_flags') or [])
        metrics = {
            'angle_deg': outcome.get('angle_deg'),
            'heading_norm': outcome.get('heading_norm'),
            'target_norm': outcome.get('target_norm'),
        }

    elif operation == 'first_visible_order':
        intervals = _aux(inputs, points_by_role, 'time_intervals') or {}
        ordered = sorted(intervals.items(), key=lambda item: item[1] if item[1] is not None else float('inf'))
        value = [name for name, _ in ordered]
        unit = 'option'
        metrics = {'ordered': value}

    elif operation == 'room_area':
        room_points = to_points(points_by_role.get('room_region'))
        if len(room_points) < 3:
            value = None
            quality_flags.append('empty_pointcloud')
        else:
            centered = room_points - room_points.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[:2]
            projected = centered @ basis.T
            hull = _convex_hull_area(projected)
            value = round(float(hull), 4)

    elif operation == 'route_turns':
        turns = _aux(inputs, points_by_role, 'turns') or []
        value = list(turns)
        unit = 'option'
        metrics = {'turns': value}

    else:  # pragma: no cover - guarded by OPERATION_SPECS
        raise ValueError(f'Unsupported operation: {operation!r}')

    if value is None:
        if 'empty_pointcloud' not in quality_flags:
            quality_flags.append('empty_pointcloud')
    else:
        _flag_non_finite(value, quality_flags)

    if operation_id is None:
        operation_id = f'{operation}'
    evidence_ids = [
        f'{role}:{bindings.get(role, "?")}'
        for role in sorted(points_by_role)
        if role not in ('candidate_entities', 'tracks', 'time_intervals', 'room_region')
    ]
    result = OperationResult(
        operation_id=operation_id,
        operation=operation,
        constraint_version=int(data.get('version', 1) or 1),
        unit=unit,
        value=value,
        entity_bindings=bindings,
        input_evidence_ids=evidence_ids,
        coordinate_frame_id=context.get('coordinate_frame_id'),
        geometry_version=context.get('geometry_version'),
        quality_flags=sorted(set(quality_flags)),
        metrics=metrics,
        options=options,
        stability=stability,
    )
    return result.to_dict()


def _flag_non_finite(value, quality_flags: List[str]):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            quality_flags.append('non_finite_value')


def _convex_hull_area(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # 2D hull "volume" is its area
    except Exception:
        return 0.0


__all__ = [
    'OperationResult',
    'OperationResultStore',
    'execute_operation',
    'surface_distance_m',
    'percentile_extent',
    'raw_extent',
    'obb_extent',
    'relative_direction',
    'to_points',
    'centroid',
    '_pointcloud_digest',
]
