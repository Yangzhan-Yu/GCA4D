"""Fixed evidence-collection parameters, derived from the task constraint.

The Planner is a text model that picks *which* tool to call.  It has no basis
for choosing how many frames a reconstruction needs, how wide the sampling
window should be, or which detector to run - yet it was being asked to supply
all of those, and answered differently every run.  The result was that the same
question collected different frames and returned different measurements:

    stove extent:  40.8 cm (1 view) / 186 cm (3 misaligned views)

The plan is explicit that the LLM selects and fills in the *operation* and the
fixed program performs the computation.  Sampling parameters belong with the
operation, so they are pinned here and the tools ignore whatever the Planner
passes for them.

This makes a run a function of the question: same question, same frames, same
answer.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from workflow.constraints.operations import OPERATION_SPECS


@dataclass
class EvidenceProfile:
    """Sampling parameters used for one question."""
    operation: str = ''
    detector: str = 'sam3'
    # Visibility scan: a coarse pass over the whole video.
    scan_stride_seconds: float = 4.0
    max_scan_frames: int = 80
    # Bridge extraction: sampled inside each visibility interval, skipping the
    # gaps where nothing is visible.  VGGT needs overlapping views, so density
    # inside an interval matters more than covering every last second of the
    # video - sampling uniformly over the whole span put consecutive frames
    # 13.5 s apart and the reconstruction fell apart.
    bridge_interval_seconds: float = 2.0
    max_bridge_frames: int = 64
    bridge_padding_seconds: float = 1.0
    # Keyframe selection.  min_support_frames is how many selected frames must
    # actually CONTAIN the target: at 2, a real run selected six frames of which
    # only three showed the chair and the rest were trajectory fillers.
    top_k_frames: int = 12
    min_support_frames: int = 8
    min_visibility: float = 0.2
    min_temporal_gap: float = 5.0
    min_joint_visibility: float = 0.25
    # Temporal neighbours around an anchor.
    neighbor_offsets_seconds: List[float] = field(
        default_factory=lambda: [-4.0, -2.0, 2.0, 4.0]
    )
    max_frames_per_anchor: int = 4
    # Frame budget for the counting tool's own sampling pass.
    count_frames: int = 12
    # Reconstruction / measurement settings.  These change the reported value
    # directly: raw vs percentile vs OBB on one real stove gave 56.8 / 48.5 /
    # 60.6 cm against a ground truth of 62, so which one runs must be an
    # experiment setting, not a per-call choice.
    resolution_level: int = 7
    extent_method: str = 'percentile'
    extent_lower_percentile: float = 5.0
    extent_upper_percentile: float = 95.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evidence_profile_for(operation: str = '') -> EvidenceProfile:
    """The profile for one operation.

    Operations that measure an object need the whole visible span sampled so
    the point cloud covers more than one surface; presence-only operations can
    be satisfied by fewer frames, but the same numbers are used for every
    operation today so that a change in profile is a deliberate change in
    experiment settings rather than an incidental one.
    """
    profile = EvidenceProfile(operation=str(operation or ''))
    if profile.operation not in OPERATION_SPECS:
        # No executable operation (unsupported question type): keep the
        # defaults, which are a reasonable general-purpose scan.
        profile.operation = ''
    return profile


__all__ = ['EvidenceProfile', 'evidence_profile_for']
