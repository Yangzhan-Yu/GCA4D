"""Constraint pipeline: validate -> execute -> validate result.

This is the glue that turns the three gap interfaces into one callable unit.
It also owns the per-question directory layout and the provenance/version
bookkeeping required by the plan so that two measurements can only be
compared when they share a coordinate frame and geometry version.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from workflow.constraints.executor import (
    OperationResultStore,
    execute_operation,
    to_points,
)
from workflow.constraints.validator import as_dict, operation_name, validate_operation, validate_result


def question_dir(scene_root, question_id) -> Path:
    path = Path(scene_root) / 'questions' / str(question_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def constraint_path(scene_root, question_id) -> Path:
    return question_dir(scene_root, question_id) / 'task_constraint.json'


def operation_results_path(scene_root, question_id) -> Path:
    return question_dir(scene_root, question_id) / 'operation_results.json'


def save_constraint(constraint, scene_root, question_id) -> Path:
    path = constraint_path(scene_root, question_id)
    if hasattr(constraint, 'save'):
        constraint.save(path)
    else:
        path.write_text(
            json.dumps(constraint, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    return path


def load_constraint(scene_root, question_id):
    path = constraint_path(scene_root, question_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def load_result_store(scene_root, question_id) -> OperationResultStore:
    return OperationResultStore(operation_results_path(scene_root, question_id))


def log_repair(scene_root, question_id, entry: Dict[str, Any]):
    path = question_dir(scene_root, question_id) / 'repair_log.jsonl'
    payload = dict(entry)
    payload.setdefault('timestamp', time.time())
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + '\n')


# --------------------------------------------------------------------------
# provenance / versions
# --------------------------------------------------------------------------
def pointcloud_path(store_root, object_id: str) -> Path:
    return Path(store_root) / 'geometry' / f'{object_id}_points.npz'


def load_pointcloud(store_root, object_id: str, metric: bool = True) -> Optional[np.ndarray]:
    candidates = []
    if metric:
        candidates.append(Path(store_root) / 'geometry' / f'{object_id}_points_metric.npz')
    candidates.append(pointcloud_path(store_root, object_id))
    for path in candidates:
        if path.exists():
            data = np.load(path)
            if 'points' in data:
                return to_points(data['points'])
    return None


def available_geometry_ids(store_root) -> List[str]:
    """Object ids that actually have a point cloud on disk.

    Not every id in the memory store has geometry: ``query_tracks`` also
    returns observation sub-tracks (``sofa_01`` ...) that are never written as
    separate point clouds.  Validation must use this list, not the store's
    object table and certainly not the bindings themselves.
    """
    geometry_dir = Path(store_root) / 'geometry'
    ids = set()
    if geometry_dir.exists():
        for path in geometry_dir.glob('*_points_metric.npz'):
            ids.add(path.name[: -len('_points_metric.npz')])
        for path in geometry_dir.glob('*_points.npz'):
            ids.add(path.name[: -len('_points.npz')])
    return sorted(ids)


def geometry_categories(store_root) -> List[str]:
    """Category names that have a point cloud on disk."""
    categories = set()
    for object_id in available_geometry_ids(store_root):
        if '_' in object_id:
            categories.add(object_id.rsplit('_', 1)[0])
        else:
            categories.add(object_id)
    return sorted(categories)


def geometry_version(store_root, object_ids: List[str]) -> str:
    """A digest that changes whenever the underlying geometry changes."""
    parts = []
    for object_id in sorted(set(object_ids)):
        path = pointcloud_path(store_root, object_id)
        metric_path = Path(store_root) / 'geometry' / f'{object_id}_points_metric.npz'
        chosen = metric_path if metric_path.exists() else path
        if not chosen.exists():
            parts.append(f'{object_id}:missing')
            continue
        stat = chosen.stat()
        parts.append(f'{object_id}:{stat.st_size}:{int(stat.st_mtime)}')
    digest = hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()[:12]
    return f'geo_{digest}'


def build_context(
    constraint,
    scene_root,
    question_id,
    store_root: Optional[Path] = None,
    bindings: Optional[Dict[str, Any]] = None,
    points: Optional[Dict[str, Any]] = None,
    metric_scale: Optional[Dict[str, Any]] = None,
    coordinate_frame_id: Optional[str] = None,
    available_object_ids: Optional[List[str]] = None,
    available_categories: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = as_dict(constraint)
    store_root = Path(store_root) if store_root is not None else Path(scene_root)
    bindings = bindings or {}
    points = points or {}
    object_ids = [
        str(value)
        for value in bindings.values()
        if isinstance(value, str) and value
    ]
    if available_object_ids is None:
        # Fall back to the binding targets.  Callers that have a store should
        # pass available_geometry_ids(store_root) instead, otherwise the
        # availability check below can never fail.
        available_object_ids = object_ids
    context: Dict[str, Any] = {
        'entity_bindings': bindings,
        'points': points,
        'metric_scale': metric_scale or {},
        'coordinate_frame_id': coordinate_frame_id or (data.get('reference_frame') or {}).get('coordinate_frame_id'),
        'geometry_version': geometry_version(store_root, object_ids),
        'available_object_ids': sorted(set(str(v) for v in available_object_ids)),
        'available_categories': sorted(
            {str(v).strip().lower() for v in (available_categories or []) if str(v).strip()}
        ),
    }
    if extra:
        context.update(extra)
    return context


# --------------------------------------------------------------------------
# main entry point used by the agent tools
# --------------------------------------------------------------------------
def run_operation(
    constraint,
    inputs: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    scene_root=None,
    question_id=None,
    persist: bool = False,
) -> Dict[str, Any]:
    """Validate, execute and re-validate one operation.

    Returns a dict with ``status`` in {``valid``, ``rejected``} plus the
    operation result and both validation reports.  Errors never raise for
    expected failure modes so the Planner can repair them.
    """
    context = dict(context or {})
    inputs = dict(inputs or {})
    if inputs.get('bindings'):
        context['entity_bindings'] = dict(inputs['bindings'])
    if inputs.get('points'):
        context['points'] = dict(inputs['points'])
    if inputs.get('metric_scale') is not None:
        context['metric_scale'] = inputs['metric_scale']

    pre = validate_operation(constraint, context)
    if not pre['valid']:
        report = {
            'status': 'rejected',
            'stage': 'validate_operation',
            'operation': operation_name(constraint),
            'operation_validation': pre,
            'result': None,
        }
        if persist and scene_root is not None and question_id is not None:
            log_repair(scene_root, question_id, {
                'stage': 'validate_operation',
                'operation': operation_name(constraint),
                'errors': pre['errors'],
            })
        return report

    operation = operation_name(constraint)
    op_id = inputs.get('operation_id')
    store = None
    if persist and scene_root is not None and question_id is not None:
        store = load_result_store(scene_root, question_id)
        if not op_id:
            op_id = store.next_operation_id(operation or 'operation')
    if not op_id:
        op_id = operation or 'operation'

    try:
        result = execute_operation(constraint, inputs, context=context, operation_id=op_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the Planner
        return {
            'status': 'rejected',
            'stage': 'execute_operation',
            'operation': operation,
            'operation_validation': pre,
            'error': f'{type(exc).__name__}: {exc}',
            'result': None,
        }

    post = validate_result(constraint, result, context)
    result['verification_status'] = 'verified' if post['valid'] else 'rejected'
    result['verification'] = post
    if persist and store is not None:
        store.add(result)
    report = {
        'status': 'valid' if post['valid'] else 'rejected',
        'stage': 'validate_result' if not post['valid'] else 'complete',
        'operation': operation,
        'operation_id': result['operation_id'],
        'operation_validation': pre,
        'result_validation': post,
        'result': result,
    }
    if persist and scene_root is not None and question_id is not None and not post['valid']:
        log_repair(scene_root, question_id, {
            'stage': 'validate_result',
            'operation': operation,
            'operation_id': result['operation_id'],
            'errors': post['errors'],
        })
    return report


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


__all__ = [
    'run_operation',
    'build_context',
    'question_dir',
    'constraint_path',
    'operation_results_path',
    'save_constraint',
    'load_constraint',
    'load_result_store',
    'log_repair',
    'load_pointcloud',
    'geometry_version',
    'available_geometry_ids',
    'geometry_categories',
]
