import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.apis.four_d_memory import SpatialFact, open_vsibench_memory
from tools.apis.metric_scale_estimator import (
    compute_closest_point_distance,
    estimate_metric_scale,
    scale_point_cloud,
)


def parse_args():
    parser = argparse.ArgumentParser('Estimate metric scale and closest-point distance')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--question-id', type=int, required=True)
    parser.add_argument('--data-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench')
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resolution-level', type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
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
    geometry_path = question_evidence['geometry_path']
    transform_path = question_evidence['transform_path']
    evidence_dir = scene_root / 'evidence'

    frames = store.query_frames()
    result = estimate_metric_scale(
        frames=frames,
        vggt_geometry_path=geometry_path,
        vggt_transform_path=transform_path,
        device=args.device,
        resolution_level=args.resolution_level,
    )

    object_points = {}
    object_colors = {}
    for obj in store.query_objects():
        path = store.root_dir / 'geometry' / f'{obj.object_id}_points.npz'
        if not path.exists():
            continue
        data = np.load(path)
        object_points[obj.object_id] = data['points'].astype(np.float32)
        if 'colors' in data:
            object_colors[obj.object_id] = data['colors'].astype(np.uint8)

    if len(object_points) < 2:
        raise RuntimeError('Need at least two object point clouds')

    metric_points = {
        object_id: scale_point_cloud(points, result.scale_factor)
        for object_id, points in object_points.items()
    }
    for object_id, points in metric_points.items():
        save_data = {'points': points.astype(np.float32)}
        if object_id in object_colors:
            save_data['colors'] = object_colors[object_id]
        np.savez_compressed(
            store.root_dir / 'geometry' / f'{object_id}_points_metric.npz',
            **save_data,
        )

    object_ids = sorted(metric_points.keys())
    distance = compute_closest_point_distance(
        metric_points[object_ids[0]],
        metric_points[object_ids[1]],
    )
    points_a = metric_points[object_ids[0]]
    points_b = metric_points[object_ids[1]]
    centroid_a = points_a.mean(axis=0)
    centroid_b = points_b.mean(axis=0)
    centroid_distance_m = float(np.linalg.norm(centroid_a - centroid_b))

    from scipy.spatial import cKDTree
    sample_a = points_a[::max(1, len(points_a) // 3000)]
    sample_b = points_b[::max(1, len(points_b) // 3000)]
    tree_b = cKDTree(points_b)
    nearest_b, _ = tree_b.query(sample_a, k=1)
    overlap_ratio_ab = float(np.mean(nearest_b <= 0.05))
    tree_a = cKDTree(points_a)
    nearest_a, _ = tree_a.query(sample_b, k=1)
    overlap_ratio_ba = float(np.mean(nearest_a <= 0.05))
    duplicate_mask_overlap = max(overlap_ratio_ab, overlap_ratio_ba)

    quality_flags = []
    if distance['distance_m'] < 0.05:
        quality_flags.append('degenerate_distance')
    if centroid_distance_m < 0.10:
        quality_flags.append('near_identical_pointclouds')
    if duplicate_mask_overlap >= 0.50:
        quality_flags.append('duplicate_masks_suspected')

    scale_path = evidence_dir / 'metric_scale.json'
    scale_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    distance_result = {
        'object_a': object_ids[0],
        'object_b': object_ids[1],
        'scale_factor': result.scale_factor,
        'centroid_distance_m': centroid_distance_m,
        'duplicate_mask_overlap': duplicate_mask_overlap,
        'quality_flags': quality_flags,
        **distance,
    }
    distance_path = evidence_dir / 'distance_result.json'
    distance_path.write_text(
        json.dumps(distance_result, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    fallback_colors = {object_ids[0]: 'red', object_ids[1]: 'blue'}
    for object_id, points in metric_points.items():
        sample_indices = np.arange(0, len(points), max(1, len(points) // 10000))
        sample = points[sample_indices]
        if object_id in object_colors:
            colors = object_colors[object_id][sample_indices].astype(np.float32) / 255.0
        else:
            colors = fallback_colors[object_id]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, alpha=0.9,
                   label=object_id, c=colors)
    pa = np.asarray(distance['point_a'])
    pb = np.asarray(distance['point_b'])
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], c='black', linewidth=3)
    ax.scatter(*pa, c='black', s=40)
    ax.scatter(*pb, c='black', s=40)
    ax.set_title(f'Closest distance: {distance["distance_m"]:.3f} m')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(evidence_dir / 'metric_distance.png', dpi=180)
    plt.close(fig)

    store.add_spatial_fact(SpatialFact(
        fact_id=f'q{args.question_id}_metric_scale',
        subject_id=None,
        relation='metric_scale',
        value={'scale_factor': result.scale_factor},
        evidence=[item['frame_id'] for item in result.frame_results],
        confidence=result.confidence,
        metadata={'question_id': args.question_id},
    ))
    store.export_parquet()

    question_evidence['metric_scale'] = {
        'scale_factor': result.scale_factor,
        'confidence': result.confidence,
        'metric_scale_path': str(scale_path),
        'distance_result_path': str(distance_path),
        'metric_distance_path': str(evidence_dir / 'metric_distance.png'),
    }
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({
        'scale_factor': result.scale_factor,
        'confidence': result.confidence,
        'distance': distance_result,
        'metric_scale_path': str(scale_path),
        'distance_result_path': str(distance_path),
        'metric_distance_path': str(evidence_dir / 'metric_distance.png'),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
