from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from tools.utils.mask_metrics import mask_quality


@dataclass
class ObservationQuality:
    entity: str
    frame_id: str
    mask_area_ratio: float
    mask_boundary_ratio: float
    mask_bbox_coverage: float
    point_count: int
    mask_touches_border: bool = False
    mask_bbox_border_sides: int = 0
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceStatus:
    sufficient: bool
    missing: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_mask_quality(mask: np.ndarray) -> Dict[str, float]:
    """Shape statistics for a mask (see tools/utils/mask_metrics.py)."""
    return mask_quality(mask)


class EvidenceSufficiencyChecker:
    def __init__(
        self,
        min_observations_per_entity: int = 2,
        min_points_per_entity: int = 1000,
        min_mask_area_ratio: float = 0.003,
        max_boundary_ratio: float = 0.35,
        min_bbox_coverage: float = 0.20,
        max_mask_border_sides: Optional[int] = None,
    ):
        self.min_observations_per_entity = min_observations_per_entity
        self.min_points_per_entity = min_points_per_entity
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_boundary_ratio = max_boundary_ratio
        self.min_bbox_coverage = min_bbox_coverage
        self.max_mask_border_sides = max_mask_border_sides

    def assess_observation(self, observation: ObservationQuality) -> List[str]:
        reasons = []
        if observation.mask_area_ratio < self.min_mask_area_ratio:
            reasons.append('mask_area_too_small')
        sides = int(getattr(observation, 'mask_bbox_border_sides', 0) or 0)
        if (
            self.max_mask_border_sides is not None
            and sides > int(self.max_mask_border_sides)
        ):
            reasons.append('mask_clipped_by_frame_edge')
        if observation.mask_boundary_ratio > self.max_boundary_ratio:
            reasons.append('mask_touches_image_boundary')
        if observation.mask_bbox_coverage < self.min_bbox_coverage:
            reasons.append('mask_shape_too_fragmented')
        if observation.point_count < self.min_points_per_entity:
            reasons.append('too_few_3d_points')
        return reasons

    def check(
        self,
        required_entities: List[str],
        observations_by_entity: Dict[str, List[ObservationQuality]],
        required_evidence: Optional[List[str]] = None,
    ) -> EvidenceStatus:
        required_evidence = set(required_evidence or [])
        missing: Dict[str, Dict[str, Any]] = {}
        summary: Dict[str, Any] = {}

        for entity in required_entities:
            observations = observations_by_entity.get(entity, [])
            accepted = []
            rejected = []
            for observation in observations:
                reasons = self.assess_observation(observation)
                if reasons:
                    rejected.append({'frame_id': observation.frame_id, 'reasons': reasons})
                else:
                    accepted.append(observation)

            total_points = sum(obs.point_count for obs in accepted)
            entity_missing = {}
            if len(accepted) < self.min_observations_per_entity:
                entity_missing['observations'] = {
                    'current': len(accepted),
                    'required': self.min_observations_per_entity,
                }
            if total_points < self.min_points_per_entity:
                entity_missing['points'] = {
                    'current': total_points,
                    'required': self.min_points_per_entity,
                }

            summary[entity] = {
                'accepted_observations': len(accepted),
                'rejected_observations': rejected,
                'total_points': total_points,
            }
            if entity_missing:
                missing[entity] = entity_missing

        if 'metric_scale' in required_evidence:
            # Metric scale is not supplied by four_d_memory yet, so report it explicitly.
            missing.setdefault('_global', {})['metric_scale'] = {
                'current': False,
                'required': True,
            }

        return EvidenceStatus(
            sufficient=not missing,
            missing=missing,
            summary=summary,
        )


__all__ = [
    'ObservationQuality',
    'EvidenceStatus',
    'EvidenceSufficiencyChecker',
    'evaluate_mask_quality',
]
