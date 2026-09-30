from dataclasses import asdict, dataclass, field
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    scene_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    category TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    first_seen REAL,
    last_seen REAL,
    confidence REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scene_id, object_id)
);

CREATE TABLE IF NOT EXISTS frames (
    scene_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp REAL,
    dataset TEXT,
    scene_name TEXT,
    video_path TEXT,
    frame_path TEXT,
    is_keyframe INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scene_id, frame_id)
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    object_id TEXT,
    frame_id TEXT,
    timestamp REAL,
    dataset TEXT,
    scene_name TEXT,
    video_path TEXT,
    frame_path TEXT,
    bbox_json TEXT,
    mask_path TEXT,
    position_3d_json TEXT,
    orientation_json TEXT,
    velocity_json TEXT,
    confidence REAL,
    source TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS camera_poses (
    scene_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    timestamp REAL,
    t_world_cam_json TEXT,
    intrinsic_json TEXT,
    confidence REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scene_id, frame_id)
);

CREATE TABLE IF NOT EXISTS spatial_facts (
    fact_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    subject_id TEXT,
    relation TEXT NOT NULL,
    object_id TEXT,
    time_start REAL,
    time_end REAL,
    reference_frame TEXT,
    value_json TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    label TEXT NOT NULL,
    time_start REAL,
    time_end REAL,
    frame_id TEXT,
    confidence REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_frames_scene_time
    ON frames(scene_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_scene_object_time
    ON observations(scene_id, object_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_scene_frame
    ON observations(scene_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_camera_poses_scene_time
    ON camera_poses(scene_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_spatial_facts_scene_subject_relation
    ON spatial_facts(scene_id, subject_id, relation, time_start, time_end);
CREATE INDEX IF NOT EXISTS idx_events_scene_time
    ON events(scene_id, time_start, time_end);
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(',', ':'),
        default=lambda item: item.tolist() if hasattr(item, 'tolist') else str(item),
    )


def _json_loads(value: Optional[str], default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


@dataclass
class MemoryObject:
    object_id: str
    category: str
    aliases: List[str] = field(default_factory=list)
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRecord:
    frame_id: str
    timestamp: Optional[float] = None
    dataset: Optional[str] = None
    scene_name: Optional[str] = None
    video_path: Optional[str] = None
    frame_path: Optional[str] = None
    is_keyframe: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


EVIDENCE_FRAME_ROLE = 'evidence_candidate'
VISIBILITY_SCAN_FRAME_ROLE = 'visibility_scan'
CONTEXT_FRAME_ROLE = 'context'

_LEGACY_EVIDENCE_SOURCES = {
    'find_bridge_frames',
    'find_temporal_neighbors',
    'planner_tool',
    'temporal_neighbor_search',
}


def is_evidence_candidate_frame(frame: FrameRecord) -> bool:
    """Return whether a frame may be used for question-level evidence.

    Scene memory also stores low-cost context and visibility-scan frames.  They
    are useful for temporal localization, but must not silently re-enter the
    VGGT/SAM2 evidence set.  New frames should set ``metadata['role']`` to
    ``evidence_candidate`` explicitly; the source fallback keeps older runs
    compatible.
    """
    metadata = frame.metadata or {}
    role = metadata.get('role')
    if role is not None:
        return role == EVIDENCE_FRAME_ROLE
    return metadata.get('generated_by') in _LEGACY_EVIDENCE_SOURCES


def cluster_positions_by_extent(
    positions,
    extents,
    scale: float = 1.0,
    fallback_eps: float = None,
):
    """Group observations that could be views of the same rigid object.

    Two views are merged when their centroids are closer than the LARGER of the
    two observed extents (``scale`` multiplies that).  The observed extent of a
    patch is a lower bound on the object's size, so the larger of two views is
    the better estimate of the scale to compare against; two centroids further
    apart than the object itself cannot be the same object.

    Replaces a global ``eps = 0.2 * object_scale``.  That chain - stored cloud
    -> extent -> eps -> instance count - is fragile: when the cloud became one
    partial view the estimated scale fell to 0.44 m, eps to 0.088 m, and
    observations of four tables were split into six to eleven "tables".

    Adding the two radii instead of taking the larger one is also too
    permissive - it is the "bounding spheres just touch" test, which chains
    neighbouring objects together (four tables collapsed into one).  On real
    data the larger-extent rule is stable over scale in [1.0, 1.25] and does
    not sit on a knife edge:

        scale    chairs (gt 2)    tables (gt 4)
        0.50          2               5
        0.75          2               5
        1.00          2               4
        1.25          2               4
        1.50          2               2
    """
    positions = [np.asarray(position, dtype=float) for position in positions]
    if not positions:
        return []
    radii = []
    for extent in extents:
        if extent is None:
            radii.append(0.0)
            continue
        radii.append(float(np.linalg.norm(np.asarray(extent, dtype=float))) / 2.0)

    if not any(radius > 0 for radius in radii):
        # Without a size for any observation there is no basis for the rule.
        # Degenerating to "no merge at all" is what made counting return one
        # instance per frame, so say so and fall back deliberately.
        if fallback_eps:
            print(
                '[Tracks] no per-observation extent available; falling back to '
                f'a distance threshold of {fallback_eps:.3f} m',
                flush=True,
            )
            return cluster_positions_by_distance(positions, fallback_eps)
        raise ValueError(
            'cluster_positions_by_extent requires at least one observation '
            'extent; refusing to guess (every observation would become its '
            'own instance).'
        )

    parent = list(range(len(positions)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            threshold = scale * max(radii[i], radii[j])
            if threshold <= 0:
                continue
            if float(np.linalg.norm(positions[i] - positions[j])) <= threshold:
                union(i, j)

    clusters = {}
    for index in range(len(positions)):
        clusters.setdefault(find(index), []).append(index)
    return list(clusters.values())


def cluster_positions_by_distance(positions, eps: float):
    positions = [np.asarray(position, dtype=float) for position in positions]
    if not positions:
        return []
    parent = list(range(len(positions)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if float(np.linalg.norm(positions[i] - positions[j])) <= eps:
                union(i, j)

    clusters = {}
    for index in range(len(positions)):
        clusters.setdefault(find(index), []).append(index)
    return list(clusters.values())


@dataclass
class Observation:
    observation_id: str
    object_id: Optional[str] = None
    frame_id: Optional[str] = None
    timestamp: Optional[float] = None
    dataset: Optional[str] = None
    scene_name: Optional[str] = None
    video_path: Optional[str] = None
    frame_path: Optional[str] = None
    bbox: Optional[Sequence[float]] = None
    mask_path: Optional[str] = None
    position_3d: Optional[Sequence[float]] = None
    orientation: Optional[Sequence[float]] = None
    velocity: Optional[Sequence[float]] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraPose:
    frame_id: str
    timestamp: Optional[float] = None
    t_world_cam: Optional[Sequence[Sequence[float]]] = None
    intrinsic: Optional[Sequence[Sequence[float]]] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialFact:
    fact_id: str
    subject_id: Optional[str]
    relation: str
    object_id: Optional[str] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    reference_frame: Optional[str] = None
    value: Any = None
    evidence: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    event_id: str
    event_type: str
    label: str
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    frame_id: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FourDMemoryStore:
    """Scene-scoped persistent memory backed by SQLite and Parquet exports."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        root_dir: Union[str, Path],
        scene_id: str,
        dataset: Optional[str] = None,
        scene_name: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.scene_id = scene_id
        self.dataset = dataset
        self.scene_name = scene_name
        self.db_path = self.root_dir / 'memory.sqlite'
        self.parquet_dir = self.root_dir / 'parquet'
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self):
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(f'PRAGMA user_version = {self.SCHEMA_VERSION}')

    def upsert_object(self, record: MemoryObject):
        with self._connect() as connection:
            existing = connection.execute(
                'SELECT first_seen FROM objects WHERE scene_id = ? AND object_id = ?',
                (self.scene_id, record.object_id),
            ).fetchone()

            first_seen = record.first_seen
            if existing and existing['first_seen'] is not None:
                first_seen = min(existing['first_seen'], first_seen) \
                    if first_seen is not None else existing['first_seen']

            connection.execute(
                """
                INSERT INTO objects (
                    scene_id, object_id, category, aliases_json,
                    first_seen, last_seen, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scene_id, object_id) DO UPDATE SET
                    category = excluded.category,
                    aliases_json = excluded.aliases_json,
                    first_seen = excluded.first_seen,
                    last_seen = excluded.last_seen,
                    confidence = excluded.confidence,
                    metadata_json = excluded.metadata_json
                """,
                (
                    self.scene_id,
                    record.object_id,
                    record.category,
                    _json_dumps(record.aliases),
                    first_seen,
                    record.last_seen,
                    record.confidence,
                    _json_dumps(record.metadata),
                ),
            )

    def add_frame(self, record: FrameRecord):
        with self._connect() as connection:
            existing = connection.execute(
                'SELECT metadata_json FROM frames WHERE scene_id = ? AND frame_id = ?',
                (self.scene_id, record.frame_id),
            ).fetchone()
            metadata = _json_loads(
                existing['metadata_json'], {}
            ) if existing else {}
            metadata.update(record.metadata or {})
            connection.execute(
                """
                INSERT OR REPLACE INTO frames (
                    scene_id, frame_id, timestamp, dataset, scene_name,
                    video_path, frame_path, is_keyframe, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.scene_id,
                    record.frame_id,
                    record.timestamp,
                    record.dataset or self.dataset,
                    record.scene_name or self.scene_name,
                    record.video_path,
                    record.frame_path,
                    1 if record.is_keyframe else 0,
                    _json_dumps(metadata),
                ),
            )

    def add_observation(self, record: Observation):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO observations (
                    observation_id, scene_id, object_id, frame_id, timestamp,
                    dataset, scene_name, video_path, frame_path, bbox_json,
                    mask_path, position_3d_json, orientation_json, velocity_json,
                    confidence, source, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.observation_id,
                    self.scene_id,
                    record.object_id,
                    record.frame_id,
                    record.timestamp,
                    record.dataset or self.dataset,
                    record.scene_name or self.scene_name,
                    record.video_path,
                    record.frame_path,
                    _json_dumps(record.bbox) if record.bbox is not None else None,
                    record.mask_path,
                    _json_dumps(record.position_3d) if record.position_3d is not None else None,
                    _json_dumps(record.orientation) if record.orientation is not None else None,
                    _json_dumps(record.velocity) if record.velocity is not None else None,
                    record.confidence,
                    record.source,
                    _json_dumps(record.metadata),
                ),
            )

    def add_camera_pose(self, record: CameraPose):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO camera_poses (
                    scene_id, frame_id, timestamp, t_world_cam_json,
                    intrinsic_json, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.scene_id,
                    record.frame_id,
                    record.timestamp,
                    _json_dumps(record.t_world_cam) if record.t_world_cam is not None else None,
                    _json_dumps(record.intrinsic) if record.intrinsic is not None else None,
                    record.confidence,
                    _json_dumps(record.metadata),
                ),
            )

    def add_spatial_fact(self, record: SpatialFact):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO spatial_facts (
                    fact_id, scene_id, subject_id, relation, object_id,
                    time_start, time_end, reference_frame, value_json,
                    evidence_json, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.fact_id,
                    self.scene_id,
                    record.subject_id,
                    record.relation,
                    record.object_id,
                    record.time_start,
                    record.time_end,
                    record.reference_frame,
                    _json_dumps(record.value) if record.value is not None else None,
                    _json_dumps(record.evidence),
                    record.confidence,
                    _json_dumps(record.metadata),
                ),
            )

    def add_event(self, record: Event):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO events (
                    event_id, scene_id, event_type, label, time_start,
                    time_end, frame_id, confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    self.scene_id,
                    record.event_type,
                    record.label,
                    record.time_start,
                    record.time_end,
                    record.frame_id,
                    record.confidence,
                    _json_dumps(record.metadata),
                ),
            )

    def get_object(self, object_id: str) -> Optional[MemoryObject]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM objects WHERE scene_id = ? AND object_id = ?',
                (self.scene_id, object_id),
            ).fetchone()
        return self._row_to_object(row) if row else None

    def query_objects(
        self,
        category: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> List[MemoryObject]:
        clauses = ['scene_id = ?']
        params: List[Any] = [self.scene_id]

        if category is not None:
            clauses.append('category = ?')
            params.append(category)

        if alias is not None:
            clauses.append('aliases_json LIKE ?')
            params.append(f'%{alias}%')

        query = f"SELECT * FROM objects WHERE {' AND '.join(clauses)} ORDER BY object_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_object(row) for row in rows]

    def query_frames(
        self,
        time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[FrameRecord]:
        clauses = ['scene_id = ?']
        params: List[Any] = [self.scene_id]
        if time_range is not None:
            start, end = time_range
            if start is not None:
                clauses.append('timestamp >= ?')
                params.append(start)
            if end is not None:
                clauses.append('timestamp <= ?')
                params.append(end)
        query = (
            f"SELECT * FROM frames WHERE {' AND '.join(clauses)} "
            'ORDER BY timestamp, frame_id'
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_frame(row) for row in rows]

    def query_observations(
        self,
        object_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        frame_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
    ) -> List[Observation]:
        clauses = ['scene_id = ?']
        params: List[Any] = [self.scene_id]

        if object_id is not None:
            clauses.append('object_id = ?')
            params.append(object_id)
        if frame_id is not None:
            clauses.append('frame_id = ?')
            params.append(frame_id)
        if time_range is not None:
            start, end = time_range
            if start is not None:
                clauses.append('timestamp >= ?')
                params.append(start)
            if end is not None:
                clauses.append('timestamp <= ?')
                params.append(end)
        if frame_range is not None:
            start, end = frame_range
            if start is not None:
                clauses.append('frame_id >= ?')
                params.append(start)
            if end is not None:
                clauses.append('frame_id <= ?')
                params.append(end)

        query = (
            f"SELECT * FROM observations WHERE {' AND '.join(clauses)} "
            'ORDER BY timestamp, frame_id, object_id'
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def query_tracks(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        objects = {
            obj.object_id: obj for obj in self.query_objects()
        }
        tracks: Dict[str, Dict[str, Any]] = {}
        for observation in self.query_observations():
            obj = objects.get(observation.object_id)
            track_category = obj.category if obj else None
            if category is not None and track_category != category:
                continue
            track_id = (observation.metadata or {}).get('track_id')
            explicit_track = bool(track_id)
            track_id = track_id or observation.object_id or observation.observation_id
            track = tracks.setdefault(track_id, {
                'track_id': track_id,
                'object_id': observation.object_id,
                'category': track_category,
                'explicit_track': explicit_track,
                'frame_ids': [],
                'timestamps': [],
                'positions': [],
                'point_counts': [],
                'confidences': [],
                'observation_ids': [],
            })
            track['explicit_track'] = track['explicit_track'] or explicit_track
            track['frame_ids'].append(observation.frame_id)
            track['timestamps'].append(observation.timestamp)
            track['positions'].append(observation.position_3d)
            track['point_counts'].append(
                int((observation.metadata or {}).get('point_count', 0))
            )
            track['confidences'].append(observation.confidence)
            track['observation_ids'].append(observation.observation_id)

        output = []
        for track in tracks.values():
            valid_positions = [
                position for position in track['positions']
                if position is not None
            ]
            if valid_positions:
                track['centroid'] = np.mean(
                    np.asarray(valid_positions, dtype=float), axis=0
                ).astype(float).tolist()
            else:
                track['centroid'] = None
            track['total_points'] = sum(track['point_counts'])
            valid_confidences = [
                float(value) for value in track['confidences']
                if value is not None
            ]
            track['mean_confidence'] = (
                float(np.mean(valid_confidences))
                if valid_confidences else None
            )
            output.append(track)
        return output

    def get_camera_pose(self, frame_id: Optional[str] = None, timestamp: Optional[float] = None) -> Optional[CameraPose]:
        if frame_id is not None:
            query = 'SELECT * FROM camera_poses WHERE scene_id = ? AND frame_id = ?'
            params = (self.scene_id, frame_id)
        elif timestamp is not None:
            query = (
                'SELECT *, ABS(timestamp - ?) AS distance FROM camera_poses '
                'WHERE scene_id = ? ORDER BY distance LIMIT 1'
            )
            params = (timestamp, self.scene_id)
        else:
            query = 'SELECT * FROM camera_poses WHERE scene_id = ? ORDER BY timestamp LIMIT 1'
            params = (self.scene_id,)

        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._row_to_camera_pose(row) if row else None

    def query_spatial_facts(
        self,
        subject_id: Optional[str] = None,
        relation: Optional[str] = None,
        object_id: Optional[str] = None,
        time: Optional[float] = None,
    ) -> List[SpatialFact]:
        clauses = ['scene_id = ?']
        params: List[Any] = [self.scene_id]

        if subject_id is not None:
            clauses.append('subject_id = ?')
            params.append(subject_id)
        if relation is not None:
            clauses.append('relation = ?')
            params.append(relation)
        if object_id is not None:
            clauses.append('object_id = ?')
            params.append(object_id)
        if time is not None:
            clauses.append('(time_start IS NULL OR time_start <= ?)')
            params.append(time)
            clauses.append('(time_end IS NULL OR time_end >= ?)')
            params.append(time)

        query = (
            f"SELECT * FROM spatial_facts WHERE {' AND '.join(clauses)} "
            'ORDER BY time_start, relation, subject_id'
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_spatial_fact(row) for row in rows]

    def query_events(
        self,
        event_type: Optional[str] = None,
        time: Optional[float] = None,
    ) -> List[Event]:
        clauses = ['scene_id = ?']
        params: List[Any] = [self.scene_id]

        if event_type is not None:
            clauses.append('event_type = ?')
            params.append(event_type)
        if time is not None:
            clauses.append('(time_start IS NULL OR time_start <= ?)')
            params.append(time)
            clauses.append('(time_end IS NULL OR time_end >= ?)')
            params.append(time)

        query = (
            f"SELECT * FROM events WHERE {' AND '.join(clauses)} "
            'ORDER BY time_start, event_type, label'
        )
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def clear_objects_and_observations(self):
        """Remove question-independent object evidence while preserving scene geometry."""
        with self._connect() as connection:
            connection.execute(
                'DELETE FROM observations WHERE scene_id = ?',
                (self.scene_id,),
            )
            connection.execute(
                'DELETE FROM objects WHERE scene_id = ?',
                (self.scene_id,),
            )
            connection.execute(
                "DELETE FROM events WHERE scene_id = ? AND event_id = 'objects_built'",
                (self.scene_id,),
            )

    def summary(self) -> Dict[str, Any]:
        tables = ['frames', 'objects', 'observations', 'camera_poses', 'spatial_facts', 'events']
        counts = {}
        with self._connect() as connection:
            for table in tables:
                counts[table] = connection.execute(
                    f'SELECT COUNT(*) FROM {table} WHERE scene_id = ?',
                    (self.scene_id,),
                ).fetchone()[0]
        return {
            'scene_id': self.scene_id,
            'dataset': self.dataset,
            'scene_name': self.scene_name,
            'db_path': str(self.db_path),
            'counts': counts,
        }

    def export_parquet(self) -> Dict[str, str]:
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        table_files = {
            'frames': 'frames.parquet',
            'objects': 'objects.parquet',
            'observations': 'observations.parquet',
            'camera_poses': 'camera_poses.parquet',
            'spatial_facts': 'spatial_facts.parquet',
            'events': 'events.parquet',
        }
        exported = {}
        with self._connect() as connection:
            for table, filename in table_files.items():
                dataframe = pd.read_sql_query(
                    f'SELECT * FROM {table} WHERE scene_id = ?',
                    connection,
                    params=(self.scene_id,),
                )
                output_path = self.parquet_dir / filename
                dataframe.to_parquet(output_path, index=False)
                exported[table] = str(output_path)
        return exported

    @staticmethod
    def _row_to_frame(row: sqlite3.Row) -> FrameRecord:
        return FrameRecord(
            frame_id=row['frame_id'],
            timestamp=row['timestamp'],
            dataset=row['dataset'],
            scene_name=row['scene_name'],
            video_path=row['video_path'],
            frame_path=row['frame_path'],
            is_keyframe=bool(row['is_keyframe']),
            metadata=_json_loads(row['metadata_json'], {}),
        )

    @staticmethod
    def _row_to_object(row: sqlite3.Row) -> MemoryObject:
        return MemoryObject(
            object_id=row['object_id'],
            category=row['category'],
            aliases=_json_loads(row['aliases_json'], []),
            first_seen=row['first_seen'],
            last_seen=row['last_seen'],
            confidence=row['confidence'],
            metadata=_json_loads(row['metadata_json'], {}),
        )

    @staticmethod
    def _row_to_observation(row: sqlite3.Row) -> Observation:
        return Observation(
            observation_id=row['observation_id'],
            object_id=row['object_id'],
            frame_id=row['frame_id'],
            timestamp=row['timestamp'],
            dataset=row['dataset'],
            scene_name=row['scene_name'],
            video_path=row['video_path'],
            frame_path=row['frame_path'],
            bbox=_json_loads(row['bbox_json']),
            mask_path=row['mask_path'],
            position_3d=_json_loads(row['position_3d_json']),
            orientation=_json_loads(row['orientation_json']),
            velocity=_json_loads(row['velocity_json']),
            confidence=row['confidence'],
            source=row['source'],
            metadata=_json_loads(row['metadata_json'], {}),
        )

    @staticmethod
    def _row_to_camera_pose(row: sqlite3.Row) -> CameraPose:
        return CameraPose(
            frame_id=row['frame_id'],
            timestamp=row['timestamp'],
            t_world_cam=_json_loads(row['t_world_cam_json']),
            intrinsic=_json_loads(row['intrinsic_json']),
            confidence=row['confidence'],
            metadata=_json_loads(row['metadata_json'], {}),
        )

    @staticmethod
    def _row_to_spatial_fact(row: sqlite3.Row) -> SpatialFact:
        return SpatialFact(
            fact_id=row['fact_id'],
            subject_id=row['subject_id'],
            relation=row['relation'],
            object_id=row['object_id'],
            time_start=row['time_start'],
            time_end=row['time_end'],
            reference_frame=row['reference_frame'],
            value=_json_loads(row['value_json']),
            evidence=_json_loads(row['evidence_json'], []),
            confidence=row['confidence'],
            metadata=_json_loads(row['metadata_json'], {}),
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row['event_id'],
            event_type=row['event_type'],
            label=row['label'],
            time_start=row['time_start'],
            time_end=row['time_end'],
            frame_id=row['frame_id'],
            confidence=row['confidence'],
            metadata=_json_loads(row['metadata_json'], {}),
        )


def open_vsibench_memory(
    data_root: Union[str, Path],
    dataset: str,
    scene_name: str,
    scene_root: Optional[Union[str, Path]] = None,
) -> FourDMemoryStore:
    scene_id = f'vsibench/{dataset}/{scene_name}'
    root_dir = (
        Path(scene_root) / 'memory'
        if scene_root is not None
        else Path(data_root) / 'memory' / dataset / scene_name
    )
    return FourDMemoryStore(
        root_dir=root_dir,
        scene_id=scene_id,
        dataset=dataset,
        scene_name=scene_name,
    )


__all__ = [
    'CONTEXT_FRAME_ROLE',
    'EVIDENCE_FRAME_ROLE',
    'FrameRecord',
    'MemoryObject',
    'Observation',
    'CameraPose',
    'SpatialFact',
    'Event',
    'FourDMemoryStore',
    'VISIBILITY_SCAN_FRAME_ROLE',
    'cluster_positions_by_distance',
    'is_evidence_candidate_frame',
    'open_vsibench_memory',
]
