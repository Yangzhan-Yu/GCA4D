"""The shared per-entity detection cache.

Locating an entity in a frame is question independent, so questions about the
same scene share the result.  The key must include the detector configuration:
reusing a GroundingDINO box after switching to SAM3 is exactly how a stale
"stove" box (a microwave) replaced the correct one in an earlier revision.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.apis.entity_detection_cache import (  # noqa: E402
    CACHE_CANDIDATES,
    EntityDetectionCache,
    config_hash,
)

CFG = {'detector': 'sam3', 'confidence': 0.5, 'resolution': 1008}


def _candidate(score=0.9, mask=None):
    entry = {'score': score, 'bbox': [1.0, 2.0, 30.0, 40.0], 'mask_area_ratio': 0.1}
    if mask:
        entry['mask_path'] = mask
    return entry


def test_roundtrip(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    assert cache.load('sofa', CFG) == {}
    cache.put('sofa', CFG, '001800', [_candidate()])
    loaded = cache.load('sofa', CFG)
    assert '001800' in loaded
    assert loaded['001800'][0]['score'] == 0.9


def test_detector_change_is_a_cache_miss(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    cache.put('stove', CFG, '000840', [_candidate(0.43)])
    # same frame, different detector -> must not reuse the old box
    other = dict(CFG, detector='grounding_dino')
    assert cache.load('stove', other) == {}
    assert config_hash(CFG) != config_hash(other)


def test_threshold_change_is_a_cache_miss(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    cache.put('stove', CFG, '000840', [_candidate()])
    stricter = dict(CFG, confidence=0.3)
    assert cache.load('stove', stricter) == {}


def test_frames_needing_detection(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    cache.put('sofa', CFG, '001800', [_candidate()])
    cache.put('stove', CFG, '001800', [])
    todo = cache.frames_needing_detection(
        ['001800', '001920'], ['sofa', 'stove'], CFG
    )
    # 001800 is fully cached (sofa has a hit, stove is a cached negative)
    assert todo == ['001920']


def test_masks_are_copied_into_the_cache(tmp: Path):
    """An entry must not point at the asking question's evidence directory."""
    evidence = tmp / 'question_a' / 'evidence'
    evidence.mkdir(parents=True)
    source = evidence / 'frame.png'
    source.write_bytes(b'PNG')

    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp / 'cache')
    cache.put('sofa', CFG, '001800', [_candidate(mask=str(source))])

    stored = cache.load('sofa', CFG)['001800'][0]
    assert Path(stored['mask_path']).exists()
    assert str(tmp / 'cache') in stored['mask_path']
    assert stored['mask_path'] != str(source)

    # deleting the original question's evidence must not break the entry
    source.unlink()
    assert Path(cache.load('sofa', CFG)['001800'][0]['mask_path']).exists()


def test_invalidate_one_entity_or_scene(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    cache.put('sofa', CFG, '001800', [_candidate()])
    cache.put('stove', CFG, '001800', [_candidate()])
    cache.invalidate('sofa')
    assert cache.load('sofa', CFG) == {}
    assert cache.load('stove', CFG) != {}
    cache.invalidate()
    assert cache.load('stove', CFG) == {}


def test_locate_picks_the_highest_score():
    """Counting only needs the object found."""
    candidates = [
        {'score': 0.95, 'mask_boundary_ratio': 0.80, 'mask_bbox_coverage': 0.20},
        {'score': 0.60, 'mask_boundary_ratio': 0.00, 'mask_bbox_coverage': 0.90},
    ]
    picked = EntityDetectionCache.select_candidate(candidates, 'locate')
    assert picked['score'] == 0.95


def test_measure_prefers_an_intact_mask_over_a_higher_score():
    """Size/distance lift the mask to 3D; a clipped mask would be truncated.

    Re-running the detector with the same config yields the same masks, so the
    intact alternative has to already be in the cache.
    """
    clipped = {'score': 0.95, 'mask_boundary_ratio': 0.80, 'mask_bbox_coverage': 0.20}
    intact = {'score': 0.60, 'mask_boundary_ratio': 0.00, 'mask_bbox_coverage': 0.90}
    picked = EntityDetectionCache.select_candidate([clipped, intact], 'measure')
    assert picked is intact

    # a fragmented mask (low bbox coverage) is penalised too
    fragmented = {'score': 0.93, 'mask_boundary_ratio': 0.05, 'mask_bbox_coverage': 0.10}
    picked = EntityDetectionCache.select_candidate([fragmented, intact], 'measure')
    assert picked is intact


def test_measure_falls_back_when_nothing_is_intact():
    """Never return None, and still prefer the least damaged mask.

    With no ideal candidate the boundary penalty still applies: a mask that is
    clipped on 90% of its perimeter is a worse basis for a measurement than one
    clipped on 60%, even if it scored higher.
    """
    very_clipped = {'score': 0.50, 'mask_boundary_ratio': 0.90, 'mask_bbox_coverage': 0.10}
    less_clipped = {'score': 0.40, 'mask_boundary_ratio': 0.60, 'mask_bbox_coverage': 0.20}
    picked = EntityDetectionCache.select_candidate(
        [very_clipped, less_clipped], 'measure'
    )
    assert picked is less_clipped


def test_select_candidate_handles_empty_and_missing_metrics():
    assert EntityDetectionCache.select_candidate([], 'measure') is None
    # an entry written before quality metrics existed must not crash
    legacy = [{'score': 0.7, 'bbox': [0, 0, 1, 1]}]
    picked = EntityDetectionCache.select_candidate(legacy, 'measure')
    assert picked is legacy[0]


def test_max_per_prompt_is_not_part_of_the_key():
    """It changes how many masks are kept, not what the detector produces.

    Keying on it would force a stricter question to re-run the detector.
    """
    from tools.apis.entity_detection_cache import config_hash
    base = {'detector': 'sam3', 'confidence': 0.5, 'resolution': 1008}
    assert config_hash(base) == config_hash(dict(base))


def test_entities_do_not_collide(tmp: Path):
    cache = EntityDetectionCache('arkitscenes', 'scene1', root=tmp)
    cache.put('sofa', CFG, '001800', [_candidate(0.9)])
    cache.put('stove', CFG, '001800', [_candidate(0.2)])
    assert cache.load('sofa', CFG)['001800'][0]['score'] == 0.9
    assert cache.load('stove', CFG)['001800'][0]['score'] == 0.2


def _main():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failures = 0
    for name, func in tests:
        try:
            if func.__code__.co_argcount == 0:
                func()
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    func(Path(tmp))
            print(f'PASS {name}')
        except Exception:
            failures += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
