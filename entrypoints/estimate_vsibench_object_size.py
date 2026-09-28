import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.apis.four_d_memory import SpatialFact, open_vsibench_memory
from tools.apis.metric_scale_estimator import (
    estimate_metric_scale,
    scale_point_cloud,
)


def parse_args():
    parser = argparse.ArgumentParser('Estimate object size from metric point cloud')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--question-id', type=int, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--track-id', default=None)
    parser.add_argument(
        '--method',
        choices=['raw', 'percentile', 'obb'],
        default='percentile',
    )
    parser.add_argument('--lower-percentile', type=float, default=5.0)
    parser.add_argument('--upper-percentile', type=float, default=95.0)
    parser.add_argument('--data-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench')
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution-level', type=int, default=7)
    return parser.parse_args()


def raw_extent(points: np.ndarray):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return minimum, maximum, maximum - minimum


def percentile_extent(points: np.ndarray, lower: float, upper: float):
    minimum = np.percentile(points, lower, axis=0)
    maximum = np.percentile(points, upper, axis=0)
    return minimum, maximum, maximum - minimum


def obb_extent(points: np.ndarray, lower: float, upper: float):
    if len(points) < 10:
        return raw_extent(points)[2]
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(covariance)
    projected = centered @ eigenvectors
    projected_min = np.percentile(projected, lower, axis=0)
    projected_max = np.percentile(projected, upper, axis=0)
    return np.sort(projected_max - projected_min)[::-1]


def load_points_for_track(store, object_id: str, track_id: str | None):
    observations = store.query_observations(object_id=object_id)
    if track_id is not None:
        observations = [
            observation for observation in observations
            if (observation.metadata or {}).get('track_id') == track_id
        ]
    chunks = []
    used_observations = []
    for observation in observations:
        path_value = (observation.metadata or {}).get('pointcloud_path')
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        chunks.append(np.load(path)['points'].astype(np.float32))
        used_observations.append(observation)
    if chunks:
        return np.concatenate(chunks, axis=0), used_observations, 'observation_chunks'
    return None, used_observations, 'observation_chunks'


def main():
    args = parse_args()
    entity = str(args.entity).strip().lower()
    data_root = Path(args.data_root)
    scene_root = Path(args.results_root) / args.dataset / args.scene_name
    store = open_vsibench_memory(
        data_root=data_root,
        dataset=args.dataset,
        scene_name=args.scene_name,
        scene_root=scene_root,
    )
    scene_path = store.root_dir / 'scene.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    question_evidence = scene.get('question_evidence', {}).get(str(args.question_id))
    if not question_evidence:
        raise ValueError(f'No evidence found for question {args.question_id}')

    objects = [
        obj for obj in store.query_objects()
        if obj.category.strip().lower() == entity
    ]
    if not objects:
        raise ValueError(f'No object evidence found for entity: {entity}')
    obj = objects[0]

    raw_points, used_observations, source = load_points_for_track(
        store,
        obj.object_id,
        args.track_id,
    )
    if raw_points is None:
        raw_points_path = store.root_dir / 'geometry' / f'{obj.object_id}_points.npz'
        if not raw_points_path.exists():
            raise FileNotFoundError(f'Object point cloud not found: {raw_points_path}')
        raw_points = np.load(raw_points_path)['points'].astype(np.float32)
        used_observations = store.query_observations(object_id=obj.object_id)
        source = 'combined_fallback'

    evidence_dir = scene_root / 'evidence'
    metric_scale = estimate_metric_scale(
        frames=store.query_frames(),
        vggt_geometry_path=question_evidence['geometry_path'],
        vggt_transform_path=question_evidence['transform_path'],
        device=args.device,
        resolution_level=args.resolution_level,
    )
    metric_points = scale_point_cloud(raw_points, metric_scale.scale_factor)

    raw_min, raw_max, raw_extent_m = raw_extent(metric_points)
    pct_min, pct_max, pct_extent_m = percentile_extent(
        metric_points,
        args.lower_percentile,
        args.upper_percentile,
    )
    obb_extent_m = obb_extent(
        metric_points,
        args.lower_percentile,
        args.upper_percentile,
    )
    method_extents = {
        'raw': raw_extent_m,
        'percentile': pct_extent_m,
        'obb': obb_extent_m,
    }
    chosen_extent_m = method_extents[args.method]
    longest_dimension_m = float(chosen_extent_m.max())

    quality_flags = []
    if len(used_observations) < 2:
        quality_flags.append('few_observations')
    if len(metric_points) < 3000:
        quality_flags.append('few_points')
    raw_longest = float(raw_extent_m.max())
    pct_longest = float(pct_extent_m.max())
    obb_longest = float(obb_extent_m.max())
    if pct_longest > 0 and raw_longest > 1.4 * pct_longest:
        quality_flags.append('raw_extent_outliers')
    if pct_longest > 0 and abs(obb_longest - pct_longest) / pct_longest > 0.30:
        quality_flags.append('method_disagreement')

    metric_points_path = store.root_dir / 'geometry' / f'{obj.object_id}_points_metric.npz'
    np.savez_compressed(metric_points_path, points=metric_points)
    object_size_result = {
        'object_id': obj.object_id,
        'entity': entity,
        'track_id': args.track_id,
        'requested_method': args.method,
        'source': source,
        'observation_count': len(used_observations),
        'scale_factor': metric_scale.scale_factor,
        'point_count': int(len(metric_points)),
        'raw_extent_cm': (raw_extent_m * 100.0).astype(float).tolist(),
        'percentile_extent_cm': (pct_extent_m * 100.0).astype(float).tolist(),
        'obb_extent_cm': (obb_extent_m * 100.0).astype(float).tolist(),
        'chosen_extent_cm': (chosen_extent_m * 100.0).astype(float).tolist(),
        'raw_longest_dimension_cm': raw_longest * 100.0,
        'percentile_longest_dimension_cm': pct_longest * 100.0,
        'obb_longest_dimension_cm': obb_longest * 100.0,
        'longest_dimension_m': longest_dimension_m,
        'longest_dimension_cm': float(longest_dimension_m * 100.0),
        'quality_flags': quality_flags,
        'raw_min_xyz_m': raw_min.astype(float).tolist(),
        'raw_max_xyz_m': raw_max.astype(float).tolist(),
        'percentile_min_xyz_m': pct_min.astype(float).tolist(),
        'percentile_max_xyz_m': pct_max.astype(float).tolist(),
    }

    scale_path = evidence_dir / 'metric_scale.json'
    scale_path.write_text(
        json.dumps(metric_scale.to_dict(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    size_path = evidence_dir / 'object_size_result.json'
    size_path.write_text(
        json.dumps(object_size_result, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    sample = metric_points[::max(1, len(metric_points) // 10000)]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, alpha=0.6, c='red')
    ax.set_title(
        f'{obj.object_id} {args.method} longest dimension: '
        f'{object_size_result["longest_dimension_cm"]:.1f} cm'
    )
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    try:
        ax.set_box_aspect(chosen_extent_m.tolist())
    except Exception:
        pass
    fig.tight_layout()
    figure_path = evidence_dir / 'object_size.png'
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    store.add_spatial_fact(SpatialFact(
        fact_id=f'q{args.question_id}_{obj.object_id}_size',
        subject_id=obj.object_id,
        relation='longest_dimension',
        value={
            'track_id': args.track_id,
            'requested_method': args.method,
            'chosen_extent_m': chosen_extent_m.astype(float).tolist(),
            'longest_dimension_m': longest_dimension_m,
            'quality_flags': quality_flags,
        },
        evidence=[observation.frame_id for observation in used_observations],
        confidence=metric_scale.confidence,
        metadata={'question_id': args.question_id},
    ))
    store.export_parquet()

    question_evidence['metric_scale'] = {
        'scale_factor': metric_scale.scale_factor,
        'confidence': metric_scale.confidence,
        'metric_scale_path': str(scale_path),
    }
    question_evidence['object_size'] = {
        'object_id': obj.object_id,
        'track_id': args.track_id,
        'method': args.method,
        'longest_dimension_cm': object_size_result['longest_dimension_cm'],
        'quality_flags': quality_flags,
        'object_size_path': str(size_path),
        'object_size_figure_path': str(figure_path),
        'object_size_metric_points_path': str(metric_points_path),
    }
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({
        **object_size_result,
        'metric_scale_path': str(scale_path),
        'object_size_path': str(size_path),
        'object_size_figure_path': str(figure_path),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
