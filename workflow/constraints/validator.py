"""Pre-execution and post-execution validation for task constraints.

This module implements two of the three gap interfaces described in
``GCA4D_VSI_Bench_Executable_Constraints_Plan.md``:

* ``validate_operation`` checks whether a geometric computation is *legal*
  before it runs (entity binding, reference frame, geometry version, metric
  scale, units, degeneracy).
* ``validate_result`` checks whether a computed result is *usable* (finite,
  right unit, inside the answer space, stable, free of blocking quality flags).

Both functions accept either a :class:`TaskConstraint` or its dict form and
return a structured report.  The report is deliberately machine readable so the
Planner can convert it into a targeted repair action instead of a blind retry.
"""

from typing import Any, Dict, List, Optional

import math

from workflow.constraints.operations import OPERATION_SPECS


# Roles that must be bound before an operation may execute.
REQUIRED_ROLES: Dict[str, List[str]] = {
    'relative_direction': ['origin', 'forward', 'target'],
    'surface_distance': ['entity_a', 'entity_b'],
    'argmin_distance': ['reference_entity', 'candidate_entities'],
    'object_extent': ['entity'],
    'count_instances': ['category'],
    'first_visible_order': ['categories'],
    'room_area': ['room_region'],
    'route_turns': ['start', 'heading', 'landmarks'],
}

# Operations whose roles must resolve to *different* physical instances.
DISTINCT_ROLES: Dict[str, List[List[str]]] = {
    'relative_direction': [['origin', 'forward'], ['origin', 'target']],
    'surface_distance': [['entity_a', 'entity_b']],
}

# Quality flags that must block a final answer.
BLOCKING_QUALITY_FLAGS = {
    'degenerate_distance',
    'near_identical_pointclouds',
    'duplicate_masks_suspected',
    'empty_pointcloud',
    'non_finite_value',
    'scale_unresolved',
}

SUGGESTED_ACTIONS: Dict[str, List[str]] = {
    'unsupported_operation': [],
    'entity_not_bound': ['query_tracks', 'collect_question_evidence'],
    'ambiguous_entity_binding': ['detect_objects', 'select_detection', 'verify_candidate'],
    'duplicate_instance_binding': [
        'detect_objects',
        'select_detection',
        'collect_question_evidence',
    ],
    'missing_coordinate_frame': ['collect_question_evidence'],
    'inconsistent_coordinate_frame': ['collect_question_evidence'],
    'stale_geometry_version': ['collect_question_evidence'],
    'missing_metric_scale': ['estimate_metric_scale_and_distance'],
    'invalid_metric_scale': ['estimate_metric_scale_and_distance'],
    'degenerate_direction': ['find_bridge_frames', 'collect_question_evidence'],
    'empty_pointcloud': ['collect_question_evidence'],
    'unit_mismatch': [],
    'answer_not_mappable': ['validate_result'],
}


def _error(error_type: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'error_type': error_type, 'message': message}
    payload.update(extra)
    suggested = SUGGESTED_ACTIONS.get(error_type)
    if suggested:
        payload['suggested_actions'] = list(suggested)
    return payload


def as_dict(constraint: Any) -> Dict[str, Any]:
    if hasattr(constraint, 'to_dict'):
        return constraint.to_dict()
    return dict(constraint or {})


def operation_name(constraint: Any) -> Optional[str]:
    data = as_dict(constraint)
    return (data.get('operation') or {}).get('operation')


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _resolve_bindings(context: Dict[str, Any]) -> Dict[str, Any]:
    """Collect role -> instance bindings from every place the Planner may
    have recorded them."""
    bindings: Dict[str, Any] = {}
    for source in (
        context.get('entity_bindings'),
        context.get('bindings'),
    ):
        if isinstance(source, dict):
            bindings.update({str(k): v for k, v in source.items()})
    return bindings


def validate_operation(constraint: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate whether an operation may legally execute.

    Returns ``{"valid": bool, "errors": [...], "warnings": [...], ...}``.
    """
    context = context or {}
    data = as_dict(constraint)
    operation = operation_name(constraint)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    spec = OPERATION_SPECS.get(operation) if operation else None
    if spec is None:
        errors.append(_error(
            'unsupported_operation',
            f'Operation {operation!r} is not part of the executable operation set.',
            operation=operation,
        ))
        return {
            'valid': False,
            'operation': operation,
            'constraint_version': data.get('version'),
            'errors': errors,
            'warnings': warnings,
        }

    bindings = _resolve_bindings(context)
    available_ids = {str(item) for item in context.get('available_object_ids', [])}

    # 1. Entity binding -----------------------------------------------------
    resolved: Dict[str, Any] = {}
    for role in REQUIRED_ROLES.get(operation, []):
        value = bindings.get(role)
        if value in (None, '', [], {}):
            errors.append(_error(
                'entity_not_bound',
                f'Role {role!r} required by {operation} has no bound instance.',
                operation=operation,
                role=role,
            ))
            continue
        if isinstance(value, (list, tuple, set)):
            candidates = [str(item) for item in value if str(item).strip()]
            if len(candidates) == 0:
                errors.append(_error(
                    'entity_not_bound',
                    f'Role {role!r} has an empty candidate list.',
                    operation=operation,
                    role=role,
                ))
                continue
            if len(candidates) > 1:
                errors.append(_error(
                    'ambiguous_entity_binding',
                    (
                        f'Role {role!r} still has {len(candidates)} candidate '
                        'instances; disambiguate before executing.'
                    ),
                    operation=operation,
                    role=role,
                    candidates=candidates,
                ))
                continue
            resolved[role] = candidates[0]
        else:
            resolved[role] = str(value)

    # 2. Instance availability ---------------------------------------------
    if available_ids:
        for role, instance_id in resolved.items():
            if instance_id not in available_ids:
                errors.append(_error(
                    'entity_not_bound',
                    (
                        f'Role {role!r} binds to {instance_id!r}, which has no '
                        'geometry in scene memory.'
                    ),
                    operation=operation,
                    role=role,
                    instance_id=instance_id,
                ))

    # 3. Distinctness -------------------------------------------------------
    for pair in DISTINCT_ROLES.get(operation, []):
        left, right = pair
        if left in resolved and right in resolved and resolved[left] == resolved[right]:
            errors.append(_error(
                'duplicate_instance_binding',
                (
                    f'{operation} requires distinct instances but {left!r} and '
                    f'{right!r} both bind to {resolved[left]!r}.'
                ),
                operation=operation,
                entities=[resolved[left], resolved[right]],
                roles=list(pair),
            ))

    # 4. Reference frame ----------------------------------------------------
    frame_required = (
        data.get('evidence', {}).get('coordinate_frame_required', True)
        and 'coordinate_frame' in (spec.required_evidence or [])
    )
    frame_id = (
        context.get('coordinate_frame_id')
        or (context.get('coordinate_frame') or {}).get('coordinate_frame_id')
        or (data.get('reference_frame') or {}).get('coordinate_frame_id')
    )
    if frame_required and not frame_id:
        errors.append(_error(
            'missing_coordinate_frame',
            f'{operation} requires a coordinate frame id but none is available.',
            operation=operation,
        ))

    # 5. Geometry version ---------------------------------------------------
    declared_version = context.get('declared_geometry_version')
    evidence_version = context.get('geometry_version')
    if declared_version and evidence_version and declared_version != evidence_version:
        errors.append(_error(
            'stale_geometry_version',
            (
                f'Geometry version {evidence_version!r} does not match the '
                f'version {declared_version!r} recorded with the evidence.'
            ),
            operation=operation,
            expected=declared_version,
            actual=evidence_version,
        ))

    # 6. Metric scale -------------------------------------------------------
    if 'metric_scale' in (spec.required_evidence or []):
        scale = context.get('metric_scale') or {}
        factor = scale.get('scale_factor') if isinstance(scale, dict) else None
        if factor is None:
            errors.append(_error(
                'missing_metric_scale',
                f'{operation} requires a metric scale but none was estimated.',
                operation=operation,
            ))
        elif not _is_finite_number(factor) or float(factor) <= 0:
            errors.append(_error(
                'invalid_metric_scale',
                f'Metric scale factor is not a positive finite number: {factor!r}.',
                operation=operation,
                scale_factor=factor,
            ))

    # 7. Unit match ---------------------------------------------------------
    declared_unit = (data.get('operation') or {}).get('unit')
    if declared_unit and spec.unit and declared_unit != spec.unit:
        errors.append(_error(
            'unit_mismatch',
            (
                f'Constraint declares unit {declared_unit!r} but {operation} '
                f'is defined in {spec.unit!r}.'
            ),
            operation=operation,
            expected_unit=spec.unit,
            actual_unit=declared_unit,
        ))

    # 8. Direction degeneracy ----------------------------------------------
    if operation == 'relative_direction':
        for role in ('origin', 'forward', 'target'):
            points = (context.get('points') or {}).get(role)
            if points is None:
                continue
            count = len(points) if hasattr(points, '__len__') else 0
            if count == 0:
                errors.append(_error(
                    'empty_pointcloud',
                    f'Role {role!r} has an empty point cloud.',
                    operation=operation,
                    role=role,
                ))
        for flag in context.get('direction_flags', []) or []:
            if flag == 'degenerate_direction':
                errors.append(_error(
                    'degenerate_direction',
                    'Origin/forward entities are too close to define a heading.',
                    operation=operation,
                ))

    # 9. Point cloud presence ----------------------------------------------
    for role, instance_id in resolved.items():
        if role in ('category', 'categories'):
            continue
        points = (context.get('points') or {}).get(role)
        if points is not None and hasattr(points, '__len__') and len(points) == 0:
            errors.append(_error(
                'empty_pointcloud',
                f'Role {role!r} ({instance_id}) has an empty point cloud.',
                operation=operation,
                role=role,
                instance_id=instance_id,
            ))

    return {
        'valid': len(errors) == 0,
        'operation': operation,
        'constraint_version': data.get('version'),
        'resolved_bindings': resolved,
        'coordinate_frame_id': frame_id,
        'geometry_version': evidence_version or declared_version,
        'errors': errors,
        'warnings': warnings,
    }


def map_value_to_option(value, options, unit: str = '') -> Optional[str]:
    """Map a numeric value onto the closest answer option, if any.

    VSI-Bench numeric answers are compared with a relative tolerance, and the
    option text usually contains a number plus a unit.  We therefore parse the
    option strings and pick the closest numeric match when one is close enough.
    """
    if not options:
        return None
    import re

    if isinstance(value, str):
        for option in options:
            if option.strip().lower() == value.strip().lower():
                return option
        return None

    numbers: List[tuple] = []
    for option in options:
        match = re.findall(r'-?\d+(?:\.\d+)?', str(option))
        if match:
            numbers.append((float(match[0]), option))
    if not numbers:
        for option in options:
            if str(option).strip().lower() == str(value).strip().lower():
                return option
        return None

    if not _is_finite_number(value):
        return None
    target = float(value)
    best = min(numbers, key=lambda item: abs(item[0] - target))
    return best[1]


def validate_result(
    constraint: Any,
    result: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a computed operation result before it may be finalized."""
    context = context or {}
    data = as_dict(constraint)
    operation = operation_name(constraint)
    spec = OPERATION_SPECS.get(operation) if operation else None
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if spec is None:
        errors.append(_error(
            'unsupported_operation',
            f'Cannot validate result for unsupported operation {operation!r}.',
            operation=operation,
        ))
        return _result_report(operation, data, errors, warnings, None)

    value = result.get('value')
    if value is None:
        errors.append(_error(
            'non_finite_value',
            'Operation result has no value.',
            operation=operation,
        ))

    unit = result.get('unit')
    if spec.unit and unit and unit != spec.unit:
        errors.append(_error(
            'unit_mismatch',
            f'Result unit {unit!r} does not match {spec.unit!r}.',
            operation=operation,
            expected_unit=spec.unit,
            actual_unit=unit,
        ))

    flags = [str(flag) for flag in (result.get('quality_flags') or [])]
    blocking = sorted(set(flags) & BLOCKING_QUALITY_FLAGS)
    for flag in blocking:
        errors.append(_error(
            flag,
            f'Quality flag {flag!r} blocks finalizing the result.',
            operation=operation,
            quality_flag=flag,
        ))

    mapped: Optional[str] = None
    options = result.get('options') or (data.get('operation') or {}).get('options') or []
    if spec.unit == 'option':
        mapped = map_value_to_option(value, options, spec.unit)
        if options and mapped is None:
            errors.append(_error(
                'answer_not_mappable',
                (
                    f'Value {value!r} cannot be mapped onto the answer options '
                    f'{options!r}.'
                ),
                operation=operation,
                value=value,
                options=list(options),
            ))
    elif spec.unit in {'m', 'cm', 'm2', 'count'}:
        if value is not None and not _is_finite_number(value):
            errors.append(_error(
                'non_finite_value',
                f'Numeric operation {operation} produced a non-finite value {value!r}.',
                operation=operation,
                value=value,
            ))
        if _is_finite_number(value) and float(value) < 0:
            warnings.append({
                'error_type': 'negative_value',
                'message': f'{operation} produced a negative value {value!r}.',
            })

    # Optional stability requirement supplied by the operation itself.
    stability = result.get('stability') or {}
    tol = stability.get('tolerance')
    spread = stability.get('spread')
    if _is_finite_number(tol) and _is_finite_number(spread):
        if float(spread) > float(tol):
            errors.append(_error(
                'unstable_result',
                (
                    f'Result spread {spread} exceeds tolerance {tol}; collect '
                    'more consistent views before finalizing.'
                ),
                operation=operation,
                spread=float(spread),
                tolerance=float(tol),
            ))

    return _result_report(operation, data, errors, warnings, mapped)


def _result_report(
    operation: Optional[str],
    data: Dict[str, Any],
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    mapped: Optional[str],
) -> Dict[str, Any]:
    return {
        'valid': len(errors) == 0,
        'operation': operation,
        'constraint_version': data.get('version'),
        'mapped_answer': mapped,
        'errors': errors,
        'warnings': warnings,
    }


__all__ = [
    'validate_operation',
    'validate_result',
    'map_value_to_option',
    'REQUIRED_ROLES',
    'BLOCKING_QUALITY_FLAGS',
]
