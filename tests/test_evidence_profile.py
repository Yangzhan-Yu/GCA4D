"""Sampling parameters must not come from the Planner.

Evidence collection parameters decide which frames are reconstructed, and
therefore what the measurement is.  Letting a text model pick them made the
same question return different numbers on every run:

    stove extent   40.8 cm  (one view, window [8,32])
                   186 cm   (three misaligned views, wider window)

The parameters are pinned per operation now, and the tools ignore whatever the
Planner passes for them.
"""

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.constraints.evidence_profile import (  # noqa: E402
    EvidenceProfile,
    evidence_profile_for,
)

ENTRYPOINT = (
    Path(__file__).resolve().parents[1] / 'entrypoints' / 'run_vsibench_agent.py'
)

# Parameters the Planner must not be able to set through a tool call.
PINNED = [
    'detector',
    'top_k_frames',
    'min_support_frames',
    'min_visibility',
    'min_temporal_gap',
    'min_joint_visibility',
    'bridge_interval_seconds',
    'max_bridge_frames',
    'scan_stride_seconds',
    'max_scan_frames',
    'neighbor_offsets_seconds',
    'max_frames_per_anchor',
]


def _source():
    return ENTRYPOINT.read_text(encoding='utf-8')


def test_profile_has_every_pinned_parameter():
    profile = evidence_profile_for('object_extent').to_dict()
    for name in PINNED:
        assert name in profile, f'{name} missing from the profile'
    assert profile['detector'] == 'sam3'


def test_profile_is_dense_enough_for_vggt():
    """Frames must overlap; a wide stride destroys the reconstruction.

    Regression: sampling uniformly over a 169 s span with a 40-frame cap gave a
    4.5 s stride, and the selector then produced six frames 13.5 s apart.  Only
    three of them contained the target at all.
    """
    profile = evidence_profile_for('object_extent')
    assert profile.bridge_interval_seconds <= 3.0, (
        'stride too wide for VGGT to find overlap'
    )
    assert profile.max_bridge_frames >= 48, (
        'a small cap forces the stride up and starves the reconstruction'
    )
    assert profile.min_support_frames >= 6, (
        'too few selected frames are guaranteed to contain the target'
    )
    assert profile.top_k_frames >= profile.min_support_frames


def test_bridge_frames_are_sampled_inside_visibility_intervals():
    """Skip the gaps where nothing is visible instead of striding over them."""
    source = (Path(__file__).resolve().parents[1]
              / 'entrypoints' / 'run_vsibench_agent.py').read_text(encoding='utf-8')
    assert 'interval_plan' in source, (
        'find_bridge_frames should build its timestamps from the scanned '
        'visibility intervals'
    )
    assert "payload.get('intervals')" in source


def test_profile_survives_an_unknown_operation():
    profile = evidence_profile_for('not_an_operation')
    assert isinstance(profile, EvidenceProfile)
    assert profile.operation == ''
    assert profile.top_k_frames > 0


def test_tools_read_the_pinned_value_not_the_argument():
    """Every sampling read must go through pinned()."""
    source = _source()
    for name in PINNED:
        # args.get('top_k') / args['top_k'] style reads are gone
        assert f"args.get('{name}'" not in source, (
            f"{name} is still read from the Planner's arguments"
        )
        assert f"args['{name}']" not in source, (
            f"{name} is still read from the Planner's arguments"
        )


def test_sampling_parameters_are_not_in_the_tool_schemas():
    """If a parameter is not in the schema the Planner cannot attempt it."""
    source = _source()
    schema_names = [
        'top_k', 'min_support_frames', 'min_visibility', 'min_temporal_gap',
        'min_joint_visibility', 'scan_stride_seconds', 'max_scan_frames',
        'interval_seconds', 'padding_seconds', 'max_frames',
        'offsets_seconds', 'max_frames_per_anchor',
    ]
    for name in schema_names:
        assert f"'{name}': {{'type'" not in source, (
            f'{name} is still advertised to the Planner in a tool schema'
        )


def test_bridge_window_comes_from_the_scan():
    source = _source()
    assert 'suggestion.get(\'start_time\') is not None' in source, (
        'find_bridge_frames should take its window from visibility_scan.json'
    )
    assert 'respect_requested_window' in source, (
        'there must be an explicit opt-out for experiments that need one'
    )


def _main():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f'PASS {name}')
        except Exception:
            failures += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
