from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrameCandidate:
    frame_id: str
    timestamp: Optional[float] = None
    frame_path: Optional[str] = None
    visibility: Dict[str, float] = field(default_factory=dict)
    sharpness: Optional[float] = None
    camera_baseline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedFrame:
    frame_id: str
    timestamp: Optional[float]
    frame_path: Optional[str]
    score: float
    reason: str
    visibility: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_sharpness(candidates: List[FrameCandidate]) -> Dict[str, float]:
    values = [
        float(candidate.sharpness)
        for candidate in candidates
        if candidate.sharpness is not None
    ]
    if not values:
        return {candidate.frame_id: 0.0 for candidate in candidates}
    minimum, maximum = min(values), max(values)
    if maximum <= minimum:
        return {candidate.frame_id: 1.0 for candidate in candidates}
    return {
        candidate.frame_id: (
            0.0 if candidate.sharpness is None
            else (float(candidate.sharpness) - minimum) / (maximum - minimum)
        )
        for candidate in candidates
    }


def select_question_keyframes(
    candidates: List[FrameCandidate],
    required_entities: List[str],
    top_k: int = 6,
    min_temporal_gap: float = 5.0,
    min_joint_visibility: float = 0.5,
    min_support_frames: int = 2,
    intra_entity_min_gap: float = 2.0,
) -> List[SelectedFrame]:
    """Select object-first keyframes.

    Each required object gets its own keyframe budget. Joint visibility is only
    used as a soft ranking bonus; it is never required. This avoids waiting for
    a frame where every object appears together, which is unnecessary for most
    spatial questions.
    """
    if top_k <= 0:
        raise ValueError('top_k must be greater than 0')
    if not required_entities:
        raise ValueError('required_entities must not be empty')

    sharpness = _normalize_sharpness(candidates)
    selected: List[SelectedFrame] = []
    selected_ids = set()

    def is_temporally_distinct(
        candidate: FrameCandidate,
        gap: float = None,
    ) -> bool:
        gap = min_temporal_gap if gap is None else gap
        if candidate.timestamp is None:
            return True
        for item in selected:
            item_candidate = next(
                (c for c in candidates if c.frame_id == item.frame_id), None
            )
            if item_candidate is None or item_candidate.timestamp is None:
                continue
            if abs(candidate.timestamp - item_candidate.timestamp) < gap:
                return False
        return True

    def entity_support(entity: str) -> int:
        return sum(
            1 for item in selected
            if float(item.visibility.get(entity, 0.0)) > 0.0
        )

    def is_intra_entity_distinct(candidate: FrameCandidate, entity: str) -> bool:
        if candidate.timestamp is None:
            return True
        for item in selected:
            if float(item.visibility.get(entity, 0.0)) <= 0.0:
                continue
            item_candidate = next(
                (c for c in candidates if c.frame_id == item.frame_id), None
            )
            if item_candidate is None or item_candidate.timestamp is None:
                continue
            if abs(candidate.timestamp - item_candidate.timestamp) < intra_entity_min_gap:
                return False
        return True

    # Object-first pass: each entity gets its own support budget.
    support_target = max(0, min(int(min_support_frames), top_k))
    for support_round in range(support_target):
        for entity_index, entity in enumerate(required_entities):
            if len(selected) >= top_k:
                break
            if entity_support(entity) > support_round:
                continue
            entity_ranked = sorted(
                candidates,
                key=lambda candidate: (
                    float(candidate.visibility.get(entity, 0.0)),
                    sharpness.get(candidate.frame_id, 0.0),
                ),
                reverse=True,
            )
            for candidate in entity_ranked:
                if len(selected) >= top_k:
                    break
                if entity_support(entity) > support_round:
                    break
                if candidate.frame_id in selected_ids:
                    continue
                visibility = float(candidate.visibility.get(entity, 0.0))
                if visibility <= 0.0:
                    continue
                if not is_intra_entity_distinct(candidate, entity):
                    continue
                visibility_values = [
                    float(candidate.visibility.get(name, 0.0))
                    for name in required_entities
                ]
                score = (
                    0.75 * visibility
                    + 0.15 * sharpness.get(candidate.frame_id, 0.0)
                    + 0.10 * min(visibility_values)
                )
                selected.append(SelectedFrame(
                    frame_id=candidate.frame_id,
                    timestamp=candidate.timestamp,
                    frame_path=candidate.frame_path,
                    score=float(score),
                    reason=f'support_for:{entity}({visibility:.3f})',
                    visibility={
                        name: float(candidate.visibility.get(name, 0.0))
                        for name in required_entities
                    },
                    metadata=candidate.metadata,
                ))
                selected_ids.add(candidate.frame_id)

    # Fill any remaining slots. Joint visibility is only a ranking bonus here.
    for entity_index, entity in enumerate(required_entities):
        entity_ranked = sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.visibility.get(entity, 0.0)),
                sharpness.get(candidate.frame_id, 0.0),
            ),
            reverse=True,
        )
        for candidate in entity_ranked:
            if len(selected) >= top_k:
                break
            if candidate.frame_id in selected_ids:
                continue
            visibility = float(candidate.visibility.get(entity, 0.0))
            if visibility <= 0.0:
                continue
            if not is_temporally_distinct(candidate):
                continue
            visibility_values = [
                float(candidate.visibility.get(name, 0.0))
                for name in required_entities
            ]
            score = (
                0.75 * visibility
                + 0.15 * sharpness.get(candidate.frame_id, 0.0)
                + 0.10 * min(visibility_values)
            )
            selected.append(SelectedFrame(
                frame_id=candidate.frame_id,
                timestamp=candidate.timestamp,
                frame_path=candidate.frame_path,
                score=float(score),
                reason=f'fallback_best_for:{entity}({visibility:.3f})',
                visibility={
                    name: float(candidate.visibility.get(name, 0.0))
                    for name in required_entities
                },
                metadata=candidate.metadata,
            ))
            selected_ids.add(candidate.frame_id)

    selected.sort(key=lambda item: (item.timestamp is None, item.timestamp))
    return selected


def add_trajectory_bridge_frames(
    selected: List[SelectedFrame],
    candidates: List[FrameCandidate],
    max_total_frames: int = 8,
    max_gap_seconds: float = 15.0,
) -> List[SelectedFrame]:
    """Add a few zero-visibility frames to keep VGGT camera alignment continuous."""
    selected = list(selected)
    if len(selected) >= max_total_frames:
        return sorted(
            selected,
            key=lambda item: (item.timestamp is None, item.timestamp),
        )
    selected_ids = {item.frame_id for item in selected}
    ordered = sorted(
        selected,
        key=lambda item: (item.timestamp is None, item.timestamp),
    )
    additions: List[SelectedFrame] = []
    for left, right in zip(ordered, ordered[1:]):
        if left.timestamp is None or right.timestamp is None:
            continue
        if len(selected) + len(additions) >= max_total_frames:
            break
        gap = right.timestamp - left.timestamp
        if gap <= max_gap_seconds:
            continue
        bridge_count = min(2, max(1, int(gap // max_gap_seconds)))
        for bridge_index in range(1, bridge_count + 1):
            if len(selected) + len(additions) >= max_total_frames:
                break
            target_time = (
                left.timestamp
                + gap * bridge_index / (bridge_count + 1)
            )
            candidate = min(
                (
                    item for item in candidates
                    if item.timestamp is not None
                    and left.timestamp < item.timestamp < right.timestamp
                    and item.frame_id not in selected_ids
                ),
                key=lambda item: (
                    abs(item.timestamp - target_time),
                    -(item.sharpness or 0.0),
                ),
                default=None,
            )
            if candidate is None:
                continue
            additions.append(SelectedFrame(
                frame_id=candidate.frame_id,
                timestamp=candidate.timestamp,
                frame_path=candidate.frame_path,
                score=0.0,
                reason='trajectory_bridge',
                visibility={
                    entity: float(candidate.visibility.get(entity, 0.0))
                    for entity in candidate.visibility
                },
                metadata=candidate.metadata,
            ))
            selected_ids.add(candidate.frame_id)
    return sorted(
        selected + additions,
        key=lambda item: (item.timestamp is None, item.timestamp),
    )


def selected_to_dicts(selected: List[SelectedFrame]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in selected]


__all__ = [
    'FrameCandidate',
    'SelectedFrame',
    'add_trajectory_bridge_frames',
    'select_question_keyframes',
    'selected_to_dicts',
]
