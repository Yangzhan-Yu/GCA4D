from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from tools.apis.moge_model import infer as moge_infer
from tools.utils.misc import add_sys_path


MOGE_MODEL_ID = 'Ruicheng/moge-2-vitl-normal'
MOGE_ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'MoGe'



def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass
class MetricScaleResult:
    scale_factor: float
    frame_results: List[Dict]
    confidence: float

    def to_dict(self):
        return _json_safe(asdict(self))


def _load_image_tensor(frame_path: str) -> torch.Tensor:
    image = Image.open(frame_path).convert('RGB')
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def transform_depth_to_vggt(depth: torch.Tensor, transform_info: Dict) -> torch.Tensor:
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    original = depth.unsqueeze(0).float()
    preprocessed_width, preprocessed_height = transform_info['preprocessed_shape']
    resized = F.interpolate(
        original,
        size=(preprocessed_height, preprocessed_width),
        mode='bilinear',
        align_corners=False,
    )

    crop_box = transform_info.get('crop_box')
    if crop_box:
        _, crop_y1, _, crop_y2 = crop_box
        resized = resized[..., crop_y1:crop_y2, :]

    for key in ('pad_box', 'multi_image_padding'):
        padding = transform_info.get(key)
        if padding:
            resized = F.pad(resized, padding, mode='constant', value=0)

    if resized.shape[-2:] != (518, 518):
        resized = F.interpolate(
            resized, size=(518, 518), mode='bilinear', align_corners=False
        )
    return resized.squeeze(0).squeeze(0)


def estimate_metric_scale(
    frames,
    vggt_geometry_path: str,
    vggt_transform_path: str,
    device: str = 'cuda',
    resolution_level: int = 7,
):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    from moge.model.v2 import MoGeModel

    device_obj = torch.device(device)
    geometry = np.load(vggt_geometry_path, allow_pickle=False)
    frame_ids = [str(value) for value in geometry['frame_ids'].tolist()]
    transform_info = __import__('json').loads(
        Path(vggt_transform_path).read_text(encoding='utf-8')
    )
    frames_by_id = {frame.frame_id: frame for frame in frames}
    if len(frame_ids) != len(transform_info):
        raise ValueError('Geometry frame and transform count mismatch')

    model = MoGeModel.from_pretrained(
        MOGE_MODEL_ID,
        cache_dir='/data3/Agentic-Spatial-Reasoning/hf_cache/hub',
    ).to(device_obj).eval()

    frame_results = []
    ratios = []
    with torch.inference_mode():
        for index, frame_id in enumerate(frame_ids):
            if frame_id not in frames_by_id:
                continue
            frame = frames_by_id[frame_id]
            image_tensor = _load_image_tensor(frame.frame_path)
            moge_output = moge_infer(
                model,
                image_tensor,
                resolution_level=resolution_level,
                use_fp16=True,
            )
            metric_depth = moge_output['depth'].float().cpu()
            metric_mask = moge_output['mask'].cpu().bool()
            metric_depth_vggt = transform_depth_to_vggt(metric_depth, transform_info[index])
            metric_mask_vggt = transform_depth_to_vggt(
                metric_mask.float(), transform_info[index]
            ) > 0.5

            vggt_depth = torch.from_numpy(
                geometry['depth'][index].astype(np.float32)
            ).squeeze(-1)
            vggt_conf = torch.from_numpy(
                geometry['world_points_conf'][index].astype(np.float32)
            )
            valid = (
                metric_mask_vggt
                & (metric_depth_vggt > 0)
                & (vggt_depth > 0)
                & (vggt_conf >= torch.quantile(vggt_conf.flatten(), 0.5))
            )
            if valid.sum() < 100:
                frame_results.append({
                    'frame_id': frame_id,
                    'valid_pixels': int(valid.sum()),
                    'scale_factor': None,
                })
                continue

            ratio = metric_depth_vggt[valid] / vggt_depth[valid]
            ratio = ratio[torch.isfinite(ratio)]
            if ratio.numel() == 0:
                continue
            q10, q90 = torch.quantile(ratio, torch.tensor([0.1, 0.9]))
            ratio = ratio[(ratio >= q10) & (ratio <= q90)]
            scale_factor = float(torch.median(ratio))
            ratios.append(scale_factor)
            frame_results.append({
                'frame_id': frame_id,
                'valid_pixels': int(ratio.numel()),
                'scale_factor': scale_factor,
                'ratio_q10': float(q10),
                'ratio_q90': float(q90),
            })

    if not ratios:
        raise RuntimeError('Failed to estimate a valid metric scale')

    scale_factor = float(np.median(ratios))
    spread = float(np.std(ratios) / max(1e-8, scale_factor))
    confidence = float(max(0.0, min(1.0, 1.0 - spread)))

    return MetricScaleResult(
        scale_factor=scale_factor,
        frame_results=frame_results,
        confidence=confidence,
    )


def scale_point_cloud(points: np.ndarray, scale_factor: float) -> np.ndarray:
    return points.astype(np.float32) * float(scale_factor)


def compute_closest_point_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Dict:
    from scipy.spatial import cKDTree

    points_a = np.asarray(points_a, dtype=np.float32)
    points_b = np.asarray(points_b, dtype=np.float32)
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_b, index_b = tree_b.query(points_a, k=1)
    min_index_a = int(np.argmin(dist_b))
    best_a = int(min_index_a)
    best_b = int(index_b[min_index_a])
    best_distance = float(dist_b[min_index_a])

    dist_a, index_a = tree_a.query(points_b, k=1)
    min_index_b = int(np.argmin(dist_a))
    if float(dist_a[min_index_b]) < best_distance:
        best_b = int(min_index_b)
        best_a = int(index_a[min_index_b])
        best_distance = float(dist_a[min_index_b])

    return {
        'distance_m': best_distance,
        'point_a': points_a[best_a].tolist(),
        'point_b': points_b[best_b].tolist(),
        'index_a': best_a,
        'index_b': best_b,
    }


__all__ = [
    'MetricScaleResult',
    'estimate_metric_scale',
    'transform_depth_to_vggt',
    'scale_point_cloud',
    'compute_closest_point_distance',
]
