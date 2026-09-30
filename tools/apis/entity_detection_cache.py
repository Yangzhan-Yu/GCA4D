"""Per-entity detection cache, shared across questions in one scene.

Locating an entity in a frame is **question independent**::

    detect(frame_image, "sofa", detector_config) -> boxes / masks

It does not matter which question asked, so two questions about the same scene
can share the result.  That is the whole point of this cache: a question about
the sofa should not re-detect the sofa that another question already found.

What is deliberately NOT cached here: anything derived from the set of frames a
question chose - point clouds, camera poses, metric scale.  Those live in a
different reconstruction frame for every question, and mixing them would not be
"reuse", it would be silent corruption.

Cache key
---------
``(dataset, scene, entity, detector, confidence, resolution, frame_id)``

Every part matters.  Swapping the detector (say GroundingDINO for SAM3) without
changing the key is exactly how a stale box silently replaced a correct one in
an earlier revision, so the config is hashed into the filename and a mismatch
is a cache miss.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CACHE_ROOT = '.cache/entity_detection'

# Candidates kept per (entity, frame).  Counting only needs to know the object
# is there, but a size or distance question measures the object, and a mask
# clipped by the image border or a fragmented mask would corrupt that
# measurement.  Re-running the detector with the same settings reproduces the
# same masks, so the alternative has to be cached up front rather than
# produced on demand.
CACHE_CANDIDATES = 3

# What "good enough" means for a measurement, as opposed to mere presence.
MEASURE_MAX_BOUNDARY_RATIO = 0.35
MEASURE_MIN_BBOX_COVERAGE = 0.50


def cache_root() -> Path:
    """Where the shared cache lives.

    Override with ``GCA_ENTITY_CACHE``.  The default is relative to the repo so
    a fresh checkout does not collide with an old one.
    """
    root = os.environ.get('GCA_ENTITY_CACHE')
    if root:
        return Path(root)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / DEFAULT_CACHE_ROOT


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


class EntityDetectionCache:
    """Frame -> candidates, per entity, keyed by detector configuration."""

    def __init__(self, dataset: str, scene_name: str, root: Optional[Path] = None):
        self.dataset = str(dataset)
        self.scene_name = str(scene_name)
        self.root = Path(root) if root is not None else cache_root()
        self.dir = self.root / self.dataset / self.scene_name

    # ------------------------------------------------------------------ paths
    def _entity_dir(self, entity: str) -> Path:
        safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in str(entity))
        return self.dir / safe

    def _index_path(self, entity: str, config: Dict[str, Any]) -> Path:
        return self._entity_dir(entity) / f'{config_hash(config)}.json'

    def mask_dir(self, entity: str, config: Dict[str, Any]) -> Path:
        return self._entity_dir(entity) / 'masks' / config_hash(config)

    # ------------------------------------------------------------- read/write
    def load(self, entity: str, config: Dict[str, Any]) -> Dict[str, List[Dict]]:
        path = self._index_path(entity, config)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return {}
        return payload.get('frames') or {}

    def _relocate_masks(
        self,
        entity: str,
        config: Dict[str, Any],
        frame_id: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """Copy mask files into the cache so an entry is self-contained.

        The detector writes masks into the asking question's evidence
        directory.  Leaving those paths in the cache would hand a later
        question a path that belongs to (and may be deleted with) a different
        question.
        """
        target_dir = self.mask_dir(entity, config)
        relocated = []
        for index, candidate in enumerate(candidates):
            entry = dict(candidate)
            source = entry.get('mask_path')
            if source and Path(source).exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                destination = target_dir / f'{frame_id}_{index}.png'
                if not destination.exists():
                    shutil.copy2(source, destination)
                entry['mask_path'] = str(destination)
                entry['mask_source_path'] = str(source)
            relocated.append(entry)
        return relocated

    def store(
        self,
        entity: str,
        config: Dict[str, Any],
        frames: Dict[str, List[Dict]],
    ) -> None:
        if not frames:
            return
        path = self._index_path(entity, config)
        existing = self.load(entity, config)
        merged = dict(existing)
        merged.update(
            {
                frame_id: self._relocate_masks(entity, config, frame_id, candidates)
                for frame_id, candidates in frames.items()
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    'entity': entity,
                    'config': config,
                    'dataset': self.dataset,
                    'scene_name': self.scene_name,
                    'frames': merged,
                },
                ensure_ascii=False,
                indent=2,
            ) + '\n',
            encoding='utf-8',
        )

    # --------------------------------------------------------------- selection
    @staticmethod
    def select_candidate(candidates, purpose: str = 'locate'):
        """Pick a candidate for the task at hand.

        ``locate``  - counting, presence, appearance order: the object merely
                      has to be found, so the highest score wins.
        ``measure`` - size, distance, area: the mask is what gets lifted to 3D
                      and measured, so a mask cut off by the image border or a
                      fragmented one is penalised even if it scored higher.
        """
        candidates = [c for c in (candidates or []) if c]
        if not candidates:
            return None
        if purpose != 'measure':
            return max(candidates, key=lambda c: float(c.get('score') or 0.0))

        def measure_rank(candidate):
            boundary = float(candidate.get('mask_boundary_ratio') or 0.0)
            coverage = float(candidate.get('mask_bbox_coverage') or 0.0)
            score = float(candidate.get('score') or 0.0)
            intact = (
                boundary <= MEASURE_MAX_BOUNDARY_RATIO
                and coverage >= MEASURE_MIN_BBOX_COVERAGE
            )
            return (intact, score - 0.5 * boundary)

        return max(candidates, key=measure_rank)

    # ------------------------------------------------------------------ logic
    def load_for_entities(
        self,
        entities: Iterable[str],
        config: Dict[str, Any],
    ) -> Dict[str, Dict[str, List[Dict]]]:
        return {e: self.load(e, config) for e in entities}

    def frames_needing_detection(
        self,
        frame_ids: Iterable[str],
        entities: Iterable[str],
        config: Dict[str, Any],
    ) -> List[str]:
        """Frames where at least one entity has no cached result yet."""
        cached = self.load_for_entities(entities, config)
        needed = []
        for frame_id in frame_ids:
            if any(frame_id not in cached.get(e, {}) for e in entities):
                needed.append(frame_id)
        return needed

    def get(
        self,
        entity: str,
        config: Dict[str, Any],
        frame_id: str,
    ) -> Optional[List[Dict]]:
        """Candidates for one (entity, frame), or None when not cached."""
        return (self.load(entity, config) or {}).get(str(frame_id))

    def put(
        self,
        entity: str,
        config: Dict[str, Any],
        frame_id: str,
        candidates: List[Dict],
    ) -> None:
        self.store(entity, config, {str(frame_id): list(candidates or [])})

    def summarize(
        self,
        entities: Iterable[str],
        config: Dict[str, Any],
    ) -> Dict[str, int]:
        return {
            e: len(self.load(e, config)) for e in entities
        }

    # ---------------------------------------------------------- invalidation
    def invalidate(self, entity: Optional[str] = None) -> None:
        """Drop the cache for one entity, or the whole scene."""
        target = self._entity_dir(entity) if entity else self.dir
        if target.exists():
            shutil.rmtree(target)

    def summary(self) -> Dict[str, Any]:
        if not self.dir.exists():
            return {'entities': {}, 'root': str(self.dir)}
        entities = {}
        for path in sorted(self.dir.iterdir()):
            if not path.is_dir():
                continue
            entries = {}
            for index in path.glob('*.json'):
                try:
                    payload = json.loads(index.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    continue
                entries[config_hash(payload.get('config') or {})] = len(
                    payload.get('frames') or {}
                )
            entities[path.name] = entries
        return {'entities': entities, 'root': str(self.dir)}


__all__ = [
    'EntityDetectionCache',
    'cache_root',
    'config_hash',
    'DEFAULT_CACHE_ROOT',
    'CACHE_CANDIDATES',
    'MEASURE_MAX_BOUNDARY_RATIO',
    'MEASURE_MIN_BBOX_COVERAGE',
]
