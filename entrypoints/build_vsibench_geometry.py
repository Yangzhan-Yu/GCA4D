import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

from tools.apis.four_d_memory import CameraPose, open_vsibench_memory
from tools.apis.vggt_model import load_and_preprocess_images
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


CACHE_DIR = '/data3/Agentic-Spatial-Reasoning/hf_cache/hub'
MODEL_ID = 'facebook/VGGT-1B'


def parse_args():
    parser = argparse.ArgumentParser('Build VGGT geometry for a VSI-Bench scene')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument(
        '--data-root',
        default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench',
    )
    parser.add_argument('--num-geometry-frames', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def select_frames(frames, num_frames: int):
    if num_frames <= 0:
        raise ValueError('num_geometry_frames must be greater than 0')
    if len(frames) <= num_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[index] for index in indices]


def load_images(frames) -> List[Image.Image]:
    images = []
    for frame in frames:
        if not frame.frame_path or not Path(frame.frame_path).exists():
            raise FileNotFoundError(f'Frame not found: {frame.frame_path}')
        images.append(Image.open(frame.frame_path).convert('RGB'))
    return images


def build_camera_poses(
    frames,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
):
    extrinsic = extrinsic.detach().float().cpu()
    intrinsic = intrinsic.detach().float().cpu()

    if extrinsic.dim() == 4:
        extrinsic = extrinsic.squeeze(0)
    if intrinsic.dim() == 4:
        intrinsic = intrinsic.squeeze(0)

    num_frames = extrinsic.shape[0]
    if num_frames != len(frames):
        raise ValueError(
            f'Pose count mismatch: {num_frames} poses for {len(frames)} frames'
        )

    poses = []
    for index, frame in enumerate(frames):
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :4] = extrinsic[index]
        poses.append(CameraPose(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            t_world_cam=transform.tolist(),
            intrinsic=intrinsic[index].tolist(),
            confidence=None,
            metadata={
                'source': 'vggt',
                'model_id': MODEL_ID,
                'geometry_sample_index': index,
            },
        ))
    return poses


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    store = open_vsibench_memory(
        data_root=data_root,
        dataset=args.dataset,
        scene_name=args.scene_name,
        scene_root=Path(args.results_root) / args.dataset / args.scene_name,
    )

    all_frames = store.query_frames()
    frames = select_frames(all_frames, args.num_geometry_frames)

    print(f'scene: {args.dataset}/{args.scene_name}')
    print(f'memory frames: {len(all_frames)}')
    print(f'geometry frames: {len(frames)}')
    for frame in frames:
        print(f'  {frame.frame_id} t={frame.timestamp:.3f} {frame.frame_path}')

    if args.dry_run:
        print('dry_run: OK')
        return

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    device = torch.device(args.device)
    model = VGGT.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR).to(device).eval()
    images = load_images(frames)
    image_tensor, transform_info = load_and_preprocess_images(images)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == 'cuda',
    ):
        predictions = model(image_tensor)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions['pose_enc'], image_tensor.shape[-2:]
    )
    poses = build_camera_poses(frames, extrinsic, intrinsic)
    for pose in poses:
        store.add_camera_pose(pose)

    geometry_dir = store.root_dir / 'geometry'
    geometry_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = geometry_dir / 'vggt_geometry.npz'

    arrays = {
        'world_points': predictions['world_points'][0].float().cpu().numpy().astype(np.float16),
        'world_points_conf': predictions['world_points_conf'][0].float().cpu().numpy().astype(np.float16),
        'depth': predictions['depth'][0].float().cpu().numpy().astype(np.float16),
        'depth_conf': predictions['depth_conf'][0].float().cpu().numpy().astype(np.float16),
        'extrinsic': extrinsic.float().cpu().numpy(),
        'intrinsic': intrinsic.float().cpu().numpy(),
        'frame_ids': np.asarray([frame.frame_id for frame in frames]),
        'timestamps': np.asarray([frame.timestamp for frame in frames], dtype=np.float32),
    }
    np.savez_compressed(geometry_path, **arrays)

    transform_path = geometry_dir / 'vggt_transform.json'
    transform_path.write_text(
        json.dumps(transform_info, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    scene_path = store.root_dir / 'scene.json'
    scene_metadata = json.loads(scene_path.read_text(encoding='utf-8'))
    scene_metadata['geometry'] = {
        'source': 'vggt',
        'model_id': MODEL_ID,
        'geometry_path': str(geometry_path),
        'transform_path': str(transform_path),
        'frame_ids': [frame.frame_id for frame in frames],
        'num_frames': len(frames),
    }
    scene_path.write_text(
        json.dumps(scene_metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    exported = store.export_parquet()
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print(f'geometry: {geometry_path}')
    print(f'parquet camera_poses: {exported["camera_poses"]}')


if __name__ == '__main__':
    main()
