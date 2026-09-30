from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflow.constraints.operations import (
    MASK_QUALITY_REQUIREMENTS,
    OperationSpec,
    OPERATION_SPECS,
    infer_operation,
)


@dataclass
class EntityConstraint:
    role: str
    category: str
    instance_id: Optional[str] = None
    candidate_ids: List[str] = field(default_factory=list)
    status: str = 'unbound'


@dataclass
class ReferenceFrameConstraint:
    frame_type: str = 'world'
    origin_entity: Optional[str] = None
    forward_entity: Optional[str] = None
    target_entity: Optional[str] = None
    up_axis: str = 'world_up'
    coordinate_frame_id: Optional[str] = None


@dataclass
class TimeConstraint:
    mode: str = 'full_video'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationConstraint:
    operation: Optional[str]
    unit: Optional[str]
    distance_definition: Optional[str] = None
    options: List[str] = field(default_factory=list)
    answer_mapping: Dict[str, str] = field(default_factory=dict)
    required_inputs: List[str] = field(default_factory=list)


@dataclass
class EvidenceConstraint:
    required_evidence: List[str] = field(default_factory=list)
    min_observations_per_track: int = 1
    min_points_per_track: int = 0
    coordinate_frame_required: bool = True
    quality_flags: List[str] = field(default_factory=list)
    # A mask that is cut off by the image border, or fragmented, produces an
    # incomplete point cloud.  Counting only needs the object located, so it
    # sets no bar; anything that measures the object does.
    max_mask_boundary_ratio: Optional[float] = None
    min_mask_bbox_coverage: Optional[float] = None
    max_mask_border_sides: Optional[int] = None


@dataclass
class TaskConstraint:
    question_id: int
    question_type: str
    question: str
    version: int = 1
    entities: List[EntityConstraint] = field(default_factory=list)
    reference_frame: ReferenceFrameConstraint = field(default_factory=ReferenceFrameConstraint)
    time_constraint: TimeConstraint = field(default_factory=TimeConstraint)
    operation: OperationConstraint = field(default_factory=lambda: OperationConstraint(None, None))
    evidence: EvidenceConstraint = field(default_factory=EvidenceConstraint)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )


def _parse_direction_roles(question: str):
    patterns = [
        r'standing by the ([^,]+?) and facing the ([^,]+?), is the ([^ ]+)',
        r'beginning at the ([^,]+?) facing the ([^,]+?)[,.]',
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return tuple(item.strip().lower() for item in match.groups()[:3])
    return None, None, None



def _build_entities(
    operation,
    question,
    target_entities,
    reference_entities,
):
    """Assign semantic roles to entity constraints so every operation role has
    exactly one binding slot."""
    entities = []

    def add(role, category):
        category = str(category).strip().lower()
        if not category:
            return
        if any(item.role == role and item.category == category for item in entities):
            return
        entities.append(EntityConstraint(role=role, category=category))

    if operation == 'relative_direction':
        origin, forward, target = _parse_direction_roles(question)
        if target:
            add('target', target)
        else:
            for entity in target_entities:
                add('target', entity)
        if origin:
            add('origin', origin)
        if forward:
            add('forward', forward)
        for entity in reference_entities:
            if not origin:
                origin = entity
                add('origin', entity)
                continue
            if not forward:
                forward = entity
                add('forward', entity)
        return entities

    if operation == 'surface_distance':
        pool = target_entities or reference_entities
        for index, entity in enumerate(pool[:2]):
            add('entity_a' if index == 0 else 'entity_b', entity)
        for entity in pool[2:]:
            add('entity_b', entity)
        return entities

    if operation == 'argmin_distance':
        if reference_entities:
            add('reference_entity', reference_entities[0])
        for entity in target_entities:
            add('candidate_entities', entity)
        return entities

    if operation == 'object_extent':
        for entity in (target_entities or reference_entities)[:1]:
            add('entity', entity)
        return entities

    if operation == 'count_instances':
        for entity in (target_entities or reference_entities)[:1]:
            add('category', entity)
        return entities

    if operation == 'first_visible_order':
        for entity in target_entities or reference_entities:
            add('categories', entity)
        return entities

    if operation == 'room_area':
        add('room_region', 'room')
        return entities

    for entity in target_entities:
        add('target', entity)
    for entity in reference_entities:
        add('reference', entity)
    return entities

def compile_task_constraint(plan: Dict[str, Any]) -> TaskConstraint:
    question_type = str(plan.get('question_type', 'unknown'))
    question = str(plan.get('question', ''))
    target_entities = [
        str(item).strip().lower()
        for item in plan.get('target_entities', [])
        if str(item).strip()
    ]
    reference_entities = [
        str(item).strip().lower()
        for item in plan.get('reference_entities', [])
        if str(item).strip()
    ]

    operation_name = infer_operation(
        question_type,
        plan.get('relation_or_metric', ''),
    )
    operation_spec: OperationSpec = OPERATION_SPECS.get(operation_name)
    unsupported_operation = None
    if operation_spec is not None and not operation_spec.implemented:
        # Declared in the vocabulary but not executable yet: leave the
        # constraint without an operation so the Planner falls back to
        # open-ended tool use and the answer gate stays disabled.
        unsupported_operation = operation_spec.name
        operation_name = None
        operation_spec = None
    entities = _build_entities(
        operation=operation_name,
        question=question,
        target_entities=target_entities,
        reference_entities=reference_entities,
    )

    reference_frame = ReferenceFrameConstraint(
        frame_type=str(plan.get('reference_frame', 'world')),
    )
    if operation_name == 'relative_direction':
        origin, forward, target = _parse_direction_roles(question)
        reference_frame.origin_entity = origin
        reference_frame.forward_entity = forward
        reference_frame.target_entity = target
        for role, category in (
            ('origin', origin),
            ('forward', forward),
            ('target', target),
        ):
            if category and not any(item.category == category for item in entities):
                entities.append(EntityConstraint(role=role, category=category))

    required_evidence = [
        str(item) for item in plan.get('required_evidence', [])
    ]
    if operation_spec is not None:
        required_evidence = sorted(set(required_evidence) | set(operation_spec.required_evidence))
    options = list(plan.get('options') or [])
    return TaskConstraint(
        question_id=int(plan.get('question_id', -1)),
        question_type=question_type,
        question=question,
        entities=entities,
        reference_frame=reference_frame,
        time_constraint=TimeConstraint(
            mode='specified' if plan.get('time_constraint') else 'full_video',
            metadata={'raw': plan.get('time_constraint')},
        ),
        operation=OperationConstraint(
            operation=operation_name,
            unit=operation_spec.unit if operation_spec else None,
            distance_definition=(
                'closest_surface' if operation_name in {'surface_distance', 'argmin_distance'} else None
            ),
            options=options,
            required_inputs=operation_spec.required_inputs if operation_spec else [],
        ),
        evidence=EvidenceConstraint(
            required_evidence=required_evidence,
            min_observations_per_track=1,
            min_points_per_track=0,
            coordinate_frame_required=operation_name not in {'count_instances', 'first_visible_order'},
            max_mask_border_sides=(
                MASK_QUALITY_REQUIREMENTS.get(operation_name, {}).get(
                    'max_mask_border_sides'
                )
            ),
            max_mask_boundary_ratio=(
                MASK_QUALITY_REQUIREMENTS.get(operation_name, {}).get(
                    'max_mask_boundary_ratio'
                )
            ),
            min_mask_bbox_coverage=(
                MASK_QUALITY_REQUIREMENTS.get(operation_name, {}).get(
                    'min_mask_bbox_coverage'
                )
            ),
        ),
        metadata={
            'relation_or_metric': plan.get('relation_or_metric'),
            'reasoning': plan.get('reasoning'),
            'unsupported_operation': unsupported_operation,
        },
    )


__all__ = [
    'EntityConstraint',
    'ReferenceFrameConstraint',
    'TimeConstraint',
    'OperationConstraint',
    'EvidenceConstraint',
    'TaskConstraint',
    'compile_task_constraint',
]
