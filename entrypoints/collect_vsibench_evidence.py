import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from tools.apis.agent_memory import AgentMemory
from tools.apis.evidence_sufficiency import (
    EvidenceSufficiencyChecker,
    ObservationQuality,
    evaluate_mask_quality,
)
from tools.utils.mask_metrics import mask_rejection_reason
from tools.apis.entity_detection_cache import (
    CACHE_CANDIDATES,
    EntityDetectionCache,
)
from tools.apis.four_d_memory import (
    CameraPose,
    EVIDENCE_FRAME_ROLE,
    FrameRecord,
    MemoryObject,
    Observation,
    cluster_positions_by_distance,
    cluster_positions_by_extent,
    is_evidence_candidate_frame,
    open_vsibench_memory,
)
from tools.apis.temporal_neighbor_search import (
    build_neighbor_timestamps,
    extract_video_frames_at_timestamps,
)
from tools.apis.keyframe_selector import (
    FrameCandidate,
    add_trajectory_bridge_frames,
    select_question_keyframes,
    selected_to_dicts,
)
from tools.apis.vggt_model import load_and_preprocess_images
from tools.utils.mm_utils import visualize_3d_object, visualize_3d_scene


GROUNDING_DINO_ID = 'IDEA-Research/grounding-dino-base'
CACHE_DIR = '/data3/Agentic-Spatial-Reasoning/hf_cache/hub'
VGGT_MODEL_ID = 'facebook/VGGT-1B'
META_ENTITIES = {'room', 'scene', 'place', 'environment', 'area'}
SAM2_CHECKPOINT = (
    '/data3/Agentic-Spatial-Reasoning/gca-main/tools/third_party/sam2/'
    'checkpoints/sam2.1_hiera_large.pt'
)
SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


def parse_args():
    parser = argparse.ArgumentParser('Collect targeted 4D evidence for one VSI-Bench question')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--question-id', type=int, required=True)
    parser.add_argument(
        '--data-root',
        default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench',
    )
    parser.add_argument(
        '--results-root',
        default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench',
    )
    parser.add_argument('--plan', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--min-temporal-gap', type=float, default=5.0)
    parser.add_argument('--min-joint-visibility', type=float, default=0.25)
    parser.add_argument('--min-score', type=float, default=0.25)
    parser.add_argument('--min-support-frames', type=int, default=2)
    parser.add_argument('--min-visibility', type=float, default=0.2)
    parser.add_argument(
        '--detector',
        choices=['grounding_dino', 'sam3'],
        default='grounding_dino',
        help=(
            'sam3 performs text-prompted detection AND segmentation in one '
            'forward pass, replacing the grounding_dino + sam2 two-step. It '
            'runs in a separate environment (torch>=2.7) via $PYTHON_SAM3.'
        ),
    )
    parser.add_argument(
        '--sam3-python', default=None,
        help='Interpreter used for the SAM3 worker (default $PYTHON_SAM3).',
    )
    parser.add_argument('--sam3-confidence', type=float, default=0.5)
    parser.add_argument('--sam3-resolution', type=int, default=1008)
    parser.add_argument(
        '--sam3-max-per-prompt', type=int, default=1,
        help='Masks kept per prompt per frame before selection.',
    )
    parser.add_argument(
        '--mask-boundary-max', type=float, default=None,
        help=(
            'Reject observations whose mask touches the image border on more '
            'than this fraction of its perimeter. A clipped mask truncates any '
            '3D extent measured from it. Declared by the task constraint.'
        ),
    )
    parser.add_argument(
        '--max-mask-border-sides', type=int, default=None,
        help=(
            'Reject observations whose mask bounding box is clipped by the '
            'frame on more than this many sides (0-4). A truncated view gives '
            'a truncated 3D extent, but large furniture seen close up is '
            'routinely clipped, so the limit is per operation and distance '
            'sets none. Declared by the task constraint.'
        ),
    )
    parser.add_argument(
        '--mask-coverage-min', type=float, default=None,
        help=(
            'Reject observations whose mask fills less than this fraction of '
            'its own bounding box (fragmented / spill / reflection).'
        ),
    )
    parser.add_argument(
        '--refresh-detections', action='store_true',
        help=(
            'Ignore the shared per-entity detection cache and re-run the '
            'detector. Use after changing detector settings, or when a cached '
            'box looks wrong.'
        ),
    )
    parser.add_argument(
        '--entities', nargs='*', default=None,
        help=(
            "Restrict this pass to a subset of the question's entities. "
            'Defaults to every entity in the evidence plan.'
        ),
    )
    parser.add_argument('--auto-expand', action='store_true')
    parser.add_argument('--round-index', type=int, default=0)
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()



def run_sam3_batch(
    frames: Sequence,
    prompts: Sequence[str],
    work_dir: Path,
    device: str,
    confidence: float,
    resolution: int,
    max_per_prompt: int,
    python: str = None,
) -> Dict[str, Dict[str, List[Dict]]]:
    """Run the SAM3 batch worker in its own environment.

    SAM3 needs torch>=2.7 while this pipeline runs on torch 2.5, so the model
    is driven through a subprocess.  One process handles every frame, so the
    checkpoint is loaded once per evidence-collection pass.
    """
    sam3_python = python or os.environ.get('PYTHON_SAM3')
    if not sam3_python:
        raise RuntimeError(
            'The sam3 detector needs PYTHON_SAM3 (source scripts/gca_env.sh) '
            'or --sam3-python.'
        )
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    requests_path = work_dir / 'sam3_requests.json'
    output_path = work_dir / 'sam3_results.json'
    requests_path.write_text(
        json.dumps(
            {
                'prompts': [str(item) for item in prompts],
                'frames': [
                    {'frame_id': frame.frame_id, 'image_path': str(frame.frame_path)}
                    for frame in frames
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    environment = dict(os.environ)
    sam3_root = environment.get('SAM3_ROOT')
    if sam3_root:
        existing = environment.get('PYTHONPATH', '')
        environment['PYTHONPATH'] = (
            sam3_root + (os.pathsep + existing if existing else '')
        )

    command = [
        sam3_python,
        '-m',
        'entrypoints.segment_with_sam3',
        '--requests', str(requests_path),
        '--output-json', str(output_path),
        '--device', device,
        '--confidence', str(confidence),
        '--resolution', str(resolution),
        '--max-per-prompt', str(max_per_prompt),
    ]
    print(f'[SAM3] Launching: {" ".join(command)}', flush=True)
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    for line in (completed.stdout or '').splitlines():
        print(f'[SAM3] {line}', flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f'SAM3 worker failed with returncode={completed.returncode}'
        )
    payload = json.loads(output_path.read_text(encoding='utf-8'))
    results = payload.get('results') or {}
    errors = payload.get('errors') or {}
    print(
        f'[SAM3] {len(results)} frame(s) with detections, '
        f'{len(errors)} frame(s) failed',
        flush=True,
    )
    return results

def load_plan(scene_root: Path, question_id: int):
    plan_path = scene_root / 'question_plans' / f'{question_id}.json'
    if not plan_path.exists():
        raise FileNotFoundError(
            f'Evidence plan not found: {plan_path}. '
            'Run entrypoints.plan_vsibench_question first.'
        )
    return plan_path, json.loads(plan_path.read_text(encoding='utf-8'))


def unique_entities(plan: Dict) -> List[str]:
    entities = []
    for key in ('target_entities', 'reference_entities'):
        for entity in plan.get(key, []):
            entity = str(entity).strip().lower()
            if key == 'reference_entities' and entity in META_ENTITIES:
                continue
            if entity and entity not in entities:
                entities.append(entity)
    return entities


def detect_best_box_dino(model, processor, image: Image.Image, entity: str, device, threshold):
    inputs = processor(images=image, text=[entity], return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold * 0.8,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results['boxes'].detach().cpu()
    scores = results['scores'].detach().cpu()
    if boxes.numel() == 0:
        return None
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    boxes, scores = boxes[keep], scores[keep]
    if boxes.numel() == 0:
        return None
    best_index = int(torch.argmax(scores))
    return boxes[best_index].tolist(), float(scores[best_index])


def bbox_visibility(box: Sequence[float], image_size: Tuple[int, int]) -> float:
    width, height = image_size
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_fraction = area / max(1.0, float(width * height))
    return float(min(1.0, np.sqrt(area_fraction / 0.08)))


def sharpness_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_detection_overlay(image, detections, output_path):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    colors = {'sofa': (255, 80, 80), 'stove': (80, 180, 255)}
    for entity, detection in detections.items():
        x1, y1, x2, y2 = detection['bbox']
        x1 = max(0, min(width - 1, int(np.floor(x1))))
        y1 = max(0, min(height - 1, int(np.floor(y1))))
        x2 = max(0, min(width - 1, int(np.ceil(x2))))
        y2 = max(0, min(height - 1, int(np.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        color = colors.get(entity, (0, 255, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_top = max(0, y1 - 18)
        draw.rectangle((x1, text_top, min(width - 1, x1 + 160), y1), fill=color)
        draw.text((x1 + 3, text_top + 2), f"{entity}: {detection['score']:.2f}", fill=(0, 0, 0))
    overlay.save(output_path, quality=92)


def draw_mask_overlay(image, masks, output_path):
    canvas = np.asarray(image.convert('RGB')).copy()
    colors = {'sofa': np.array([255, 60, 60], dtype=np.float32),
              'stove': np.array([60, 160, 255], dtype=np.float32)}
    for entity, mask in masks.items():
        color = colors.get(entity, np.array([0, 255, 0], dtype=np.float32))
        selected = mask.astype(bool)
        canvas[selected] = (0.55 * canvas[selected] + 0.45 * color).astype(np.uint8)
    Image.fromarray(canvas).save(output_path, quality=92)


def create_selection_contact_sheet(candidates, selected_ids, output_path):
    columns = 4
    thumb_width, thumb_height = 320, 240
    label_height = 54
    rows = int(np.ceil(len(candidates) / columns))
    sheet = Image.new('RGB', (columns * thumb_width, rows * (thumb_height + label_height)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(candidates):
        image = Image.open(candidate.frame_path).convert('RGB')
        image.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x, y))
        selected = candidate.frame_id in selected_ids
        color = (0, 180, 0) if selected else (180, 0, 0)
        draw.rectangle((x, y, x + image.width, y + image.height), outline=color, width=5)
        visibility = ', '.join(
            f'{name}={candidate.visibility.get(name, 0.0):.2f}'
            for name in sorted(candidate.visibility)
        )
        label = (
            f"frame={candidate.frame_id} t={candidate.timestamp:.1f}s "
            f"{'SELECTED' if selected else 'skipped'}\n{visibility}"
        )
        draw.text((x + 5, y + thumb_height + 5), label, fill=color)
    sheet.save(output_path, quality=92)


def transform_mask_to_vggt(mask, transform_info):
    preprocessed_width, preprocessed_height = transform_info['preprocessed_shape']
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_image = mask_image.resize(
        (preprocessed_width, preprocessed_height), Image.Resampling.NEAREST
    )
    transformed = np.asarray(mask_image) > 127

    crop_box = transform_info.get('crop_box')
    if crop_box:
        _, crop_y1, _, crop_y2 = crop_box
        transformed = transformed[crop_y1:crop_y2, :]

    for key in ('pad_box', 'multi_image_padding'):
        padding = transform_info.get(key)
        if padding:
            left, right, top, bottom = padding
            transformed = np.pad(
                transformed,
                ((top, bottom), (left, right)),
                mode='constant',
                constant_values=False,
            )

    if transformed.shape != (518, 518):
        transformed = np.asarray(
            Image.fromarray((transformed.astype(np.uint8) * 255)).resize(
                (518, 518), Image.Resampling.NEAREST
            )
        ) > 127
    return transformed


def run_question_vggt(frames, device):
    images = [Image.open(frame.frame_path).convert('RGB') for frame in frames]
    model = VGGT.from_pretrained(VGGT_MODEL_ID, cache_dir=CACHE_DIR).to(device).eval()
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
    return predictions, transform_info, extrinsic, intrinsic, image_tensor.detach().cpu()


def sample_points(points, max_points=10000):
    if len(points) <= max_points:
        return points
    return points[np.linspace(0, len(points) - 1, max_points).astype(int)]


def sample_points_and_colors(points, colors=None, max_points=10000):
    if len(points) <= max_points:
        indices = np.arange(len(points))
    else:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors


def plot_object_points(object_points, output_path, object_colors=None):
    object_colors = object_colors or {}
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    fallback_colors = {'sofa': 'red', 'stove': 'blue', 'chair': 'tab:blue'}
    for entity, points in object_points.items():
        sampled_points, sampled_colors = sample_points_and_colors(
            points, object_colors.get(entity)
        )
        if sampled_colors is not None and len(sampled_colors) == len(sampled_points):
            color_values = sampled_colors.astype(np.float32) / 255.0
        else:
            color_values = fallback_colors.get(entity, 'tab:green')
        ax.scatter(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            s=2,
            alpha=0.9,
            label=entity,
            c=color_values,
        )
    ax.set_title('Target Object 3D Points')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)


def plot_object_points_gca(
    object_points,
    object_colors,
    output_path,
    max_points=30000,
):
    images = []
    for entity, points in object_points.items():
        points_np, colors_np = sample_points_and_colors(
            points,
            object_colors.get(entity),
            max_points=max_points,
        )
        if colors_np is None:
            colors_np = np.full((len(points_np), 3), 180, dtype=np.uint8)
        image = visualize_3d_object(
            torch.from_numpy(points_np.astype(np.float32)),
            torch.from_numpy(colors_np.astype(np.float32) / 255.0),
        )
        entity_path = output_path.parent / f'object_points_{entity}_gca.png'
        image.save(entity_path)
        images.append((entity, image))

    if not images:
        return
    if len(images) == 1:
        images[0][1].save(output_path)
        return

    label_height = 36
    width = max(image.width for _, image in images)
    height = sum(image.height + label_height for _, image in images)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    y = 0
    for entity, image in images:
        draw.text((8, y + 8), entity, fill='black')
        canvas.paste(image, (0, y + label_height))
        y += image.height + label_height
    canvas.save(output_path)


def plot_camera_trajectory(frames, extrinsic, output_path):
    extrinsic = extrinsic.detach().float().cpu()
    if extrinsic.dim() == 4:
        extrinsic = extrinsic.squeeze(0)
    translations = extrinsic[:, :3, 3].numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(translations[:, 0], translations[:, 2], marker='o')
    for index, point in enumerate(translations):
        ax.text(point[0], point[2], frames[index].frame_id)
    ax.set_title('Selected-frame Camera Trajectory')
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.axis('equal')
    fig.tight_layout(); fig.savefig(output_path, dpi=180); plt.close(fig)


def select_best_gpu(min_free_gb: float = 10.0):
    try:
        import pynvml
    except ImportError:
        return None

    pynvml.nvmlInit()
    try:
        best_index, best_free = None, -1
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = memory.free / (1024 ** 3)
            if free_gb > best_free:
                best_index, best_free = index, free_gb
        if best_index is not None and best_free >= min_free_gb:
            return best_index, best_free
    finally:
        pynvml.nvmlShutdown()
    return None


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    if args.plan:
        plan_path = Path(args.plan)
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        scene_root = results_root / args.dataset / args.scene_name
        plan_path, plan = load_plan(scene_root, args.question_id)
    entities = unique_entities(plan)
    if args.entities:
        requested = [str(item).strip().lower() for item in args.entities if str(item).strip()]
        missing = [item for item in requested if item not in entities]
        if missing:
            print(
                f'[Entities] {missing} are not in the evidence plan for this '
                'question; ignoring them.',
                flush=True,
            )
        selected = [item for item in requested if item in entities]
        if selected:
            entities = selected
    if not entities:
        raise ValueError('Evidence plan contains no target/reference entities')

    # How good the masks have to be depends on what the question does with
    # them: counting only needs the object located, while size/distance lift the
    # mask to 3D and measure it.  The cache is shared either way - only the
    # choice among cached candidates differs.
    MEASURE_QUESTION_TYPES = {
        'object_size_estimation',
        'object_abs_distance',
        'object_rel_distance',
        'room_size_estimation',
    }
    question_type = str(plan.get('question_type') or '').strip()
    if question_type in MEASURE_QUESTION_TYPES:
        mask_purpose = 'measure'
    elif question_type.startswith('object_rel_direction'):
        mask_purpose = 'orient'
    else:
        mask_purpose = 'locate'
    print(f'[Detect] mask purpose={mask_purpose} (question_type={question_type!r})',
          flush=True)

    store = open_vsibench_memory(
        data_root=data_root,
        dataset=args.dataset,
        scene_name=args.scene_name,
        scene_root=scene_root,
    )
    memory_frames = store.query_frames()
    all_frames = [
        frame for frame in memory_frames
        if is_evidence_candidate_frame(frame)
    ]
    frames_by_id = {frame.frame_id: frame for frame in all_frames}
    print(f'question: {args.question_id}')
    print(f'entities: {entities}')
    print(
        f'memory frames: {len(memory_frames)} '
        f'(evidence candidates: {len(all_frames)})'
    )
    print(f'candidate frames: {len(all_frames)}')

    if args.dry_run:
        print('dry_run: OK')
        return
    if not all_frames:
        raise RuntimeError(
            'No evidence-candidate frames are available. Run '
            'scan_entity_visibility and find_bridge_frames first; legacy '
            'uniform scene frames are intentionally excluded from question evidence.'
        )
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    if args.device.startswith('cuda') and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        selected_gpu = select_best_gpu(min_free_gb=10.0)
        if selected_gpu is not None:
            gpu_index, free_gb = selected_gpu
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
            print(
                f'[GPU] Selected physical GPU {gpu_index} '
                f'with {free_gb:.2f} GiB free.',
                flush=True,
            )

    device = torch.device(args.device)
    processor, detector = None, None
    if args.detector == 'grounding_dino':
        processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID, cache_dir=CACHE_DIR, local_files_only=True)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_ID, cache_dir=CACHE_DIR, local_files_only=True
        ).to(device).eval()

    detection_overrides_path = scene_root / 'evidence' / 'detection_overrides.json'
    detection_overrides = (
        json.loads(detection_overrides_path.read_text(encoding='utf-8'))
        if detection_overrides_path.exists()
        else {}
    )

    def score_frame(frame):
        image = Image.open(frame.frame_path).convert('RGB')
        metadata = frame.metadata or {}
        cached_detections = metadata.get('detections')
        cached_visibility = metadata.get('visibility')
        cached_source = metadata.get('detection_source')
        if (
            cached_detections is not None
            and cached_visibility is not None
            and cached_source == args.detector
        ):
            filtered_detections = {}
            filtered_visibility = {}
            for entity in entities:
                detection = cached_detections.get(entity)
                visibility_value = float(cached_visibility.get(entity, 0.0))
                override = detection_overrides.get(frame.frame_id, {}).get(entity)
                if override:
                    bbox = override['bbox']
                    score = float(override.get('score', 1.0))
                    filtered_detections[entity] = {
                        'bbox': bbox,
                        'score': score,
                        'verified_category': entity,
                        'source': 'planner_disambiguation',
                    }
                    filtered_visibility[entity] = (
                        bbox_visibility(bbox, image.size) * max(0.1, score)
                    )
                    continue
                if detection is None or visibility_value <= 0.0:
                    filtered_visibility[entity] = 0.0
                    continue
                bbox = detection.get('bbox')
                filtered_detections[entity] = {
                    **detection,
                    'verified_category': entity,
                }
                filtered_visibility[entity] = visibility_value
            cached_detections = filtered_detections
            cached_visibility = filtered_visibility
            print(f'using cached visibility for frame={frame.frame_id}', flush=True)
            return FrameCandidate(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                frame_path=frame.frame_path,
                visibility=cached_visibility,
                sharpness=sharpness_score(image),
                metadata={
                    'detections': cached_detections,
                    'reused_cache': True,
                },
            )
        detections = {}
        visibility = {}
        if args.detector == 'sam3':
            hits_by_entity = sam3_cache.get(frame.frame_id, {})
            for entity in entities:
                override = detection_overrides.get(frame.frame_id, {}).get(entity)
                if override:
                    bbox = override['bbox']
                    score = float(override.get('score', 1.0))
                    detections[entity] = {
                        'bbox': bbox,
                        'score': score,
                        'verified_category': entity,
                        'source': 'planner_disambiguation',
                    }
                    visibility[entity] = (
                        bbox_visibility(bbox, image.size) * max(0.1, score)
                    )
                    continue
                hits = hits_by_entity.get(entity) or []
                if not hits:
                    visibility[entity] = 0.0
                    continue
                best = EntityDetectionCache.select_candidate(hits, mask_purpose)
                if best is None:
                    visibility[entity] = 0.0
                    continue
                detections[entity] = {
                    'bbox': [float(v) for v in best['bbox']],
                    'score': float(best['score']),
                    'verified_category': entity,
                    'mask_path': best.get('mask_path'),
                    'mask_area_ratio': best.get('mask_area_ratio'),
                    'mask_boundary_ratio': best.get('mask_boundary_ratio'),
                    'mask_bbox_coverage': best.get('mask_bbox_coverage'),
                    'mask_purpose': mask_purpose,
                    'candidates_considered': len(hits),
                    'source': 'sam3',
                }
                visibility[entity] = (
                    bbox_visibility(detections[entity]['bbox'], image.size)
                    * max(0.1, float(best['score']))
                )
        else:
            raw_detections = {}
            for entity in entities:
                override = detection_overrides.get(frame.frame_id, {}).get(entity)
                if override:
                    raw_detections[entity] = {
                        'bbox': override['bbox'],
                        'score': float(override.get('score', 1.0)),
                        'verified_category': entity,
                        'source': 'planner_disambiguation',
                    }
                    continue
                print(
                    f'scoring frame={frame.frame_id} entity={entity}',
                    flush=True,
                )
                detected = detect_best_box_dino(
                    detector,
                    processor,
                    image,
                    entity,
                    device,
                    args.min_score,
                )
                if detected is None:
                    continue
                bbox, score = detected
                raw_detections[entity] = {
                    'bbox': bbox,
                    'score': score,
                    'verified_category': entity,
                }
            for entity, detection in raw_detections.items():
                bbox = detection['bbox']
                score = float(detection['score'])
                bbox_area_ratio = (
                    max(0.0, bbox[2] - bbox[0])
                    * max(0.0, bbox[3] - bbox[1])
                    / max(1.0, float(image.width * image.height))
                )
                detections[entity] = {
                    **detection,
                    'score': score,
                }
                visibility[entity] = (
                    bbox_visibility(bbox, image.size) * max(0.1, score)
                )
        return FrameCandidate(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            frame_path=frame.frame_path,
            visibility=visibility,
            sharpness=sharpness_score(image),
            metadata={'detections': detections},
        )

    mask_rejections: Dict[str, List[Dict]] = {entity: [] for entity in entities}
    sam3_cache: Dict[str, Dict[str, List[Dict]]] = {}
    if args.detector == 'sam3' and not args.selection_only and not args.dry_run:
        # Locating an entity in a frame does not depend on the question, so
        # results are shared across questions in the same scene.  The key
        # includes the detector configuration: swapping the detector without
        # changing the key is how a stale box replaced a correct one before.
        cache = EntityDetectionCache(args.dataset, args.scene_name)
        # max_per_prompt is NOT part of the key: it changes how many of the
        # masks we keep, not what the model produces.  Keying on it would make
        # a stricter question re-run the detector for results we already have.
        cache_config = {
            'detector': 'sam3',
            'confidence': float(args.sam3_confidence),
            'resolution': int(args.sam3_resolution),
        }
        keep = max(int(args.sam3_max_per_prompt), CACHE_CANDIDATES)
        if args.refresh_detections:
            for entity in entities:
                cache.invalidate(entity)
            print('[Cache] --refresh-detections: re-running every entity.',
                  flush=True)
        cached = cache.load_for_entities(entities, cache_config)
        cached_counts = {e: len(v) for e, v in cached.items()}
        if any(cached_counts.values()):
            print(f'[Cache] hit for {cached_counts} (key={cache_config})', flush=True)

        all_ids = [frame.frame_id for frame in all_frames]
        todo_ids = set()
        for entity in entities:
            hits = cached.get(entity) or {}
            todo_ids.update(fid for fid in all_ids if fid not in hits)
        todo = [f for f in all_frames if f.frame_id in todo_ids]
        if len(todo) < len(all_frames):
            print(
                f'[Cache] {len(all_frames) - len(todo)}/{len(all_frames)} '
                'frame(s) fully cached; re-running detection only where needed.',
                flush=True,
            )

        fresh: Dict[str, Dict[str, List[Dict]]] = {}
        if todo:
            fresh = run_sam3_batch(
                frames=todo,
                prompts=entities,
                work_dir=scene_root / 'evidence' / 'sam3',
                device=args.device,
                confidence=args.sam3_confidence,
                resolution=args.sam3_resolution,
                max_per_prompt=keep,
                python=args.sam3_python,
            )
            for entity in entities:
                updates = {
                    fid: (frame_result.get(entity) or [])
                    for fid, frame_result in fresh.items()
                }
                updates = {k: v for k, v in updates.items() if v}
                cache.store(entity, cache_config, updates)

        # Re-read so this pass sees cache hits and fresh results alike, and
        # refresh any frame that was not requested this round.
        for entity in entities:
            merged = dict(cached.get(entity) or {})
            for fid, frame_result in fresh.items():
                hits = frame_result.get(entity) or []
                if hits:
                    merged[fid] = hits
            for fid in all_ids:
                if fid not in merged:
                    continue
                sam3_cache.setdefault(fid, {})[entity] = merged[fid]

    if args.detector == 'sam3' and not args.selection_only and not args.dry_run:
        print(
            f'[Cache] after this pass: {cache.summarize(entities, cache_config)}',
            flush=True,
        )
        # With the entity cache, sam3_results.json only holds the frames this
        # pass had to detect.  Write what was actually fed into scoring so the
        # evidence directory stays interpretable.
        used_path = scene_root / 'evidence' / 'sam3' / 'detections_used.json'
        used_path.parent.mkdir(parents=True, exist_ok=True)
        used_path.write_text(
            json.dumps(
                {
                    'prompts': entities,
                    'mask_purpose': mask_purpose,
                    'cache_config': cache_config,
                    'frames_scored': len(sam3_cache),
                    'cache_state': cache.summarize(entities, cache_config),
                    'results': sam3_cache,
                },
                ensure_ascii=False,
                indent=2,
            ) + '\n',
            encoding='utf-8',
        )

    scored_candidates = {}
    for frame in all_frames:
        scored_candidates[frame.frame_id] = score_frame(frame)

    reused = sum(
        1 for candidate in scored_candidates.values()
        if (candidate.metadata or {}).get('reused_cache')
    )
    if reused:
        print(
            f'[Intermediate] reused {reused}/{len(scored_candidates)} frame(s) '
            f'with cached {args.detector} detections.',
            flush=True,
        )

    candidates = [scored_candidates[frame.frame_id] for frame in all_frames]
    for candidate in candidates:
        frame = frames_by_id.get(candidate.frame_id)
        if frame is None:
            continue
        metadata = dict(frame.metadata or {})
        metadata['detections'] = candidate.metadata.get('detections', {})
        metadata['visibility'] = candidate.visibility
        metadata['detection_cache_version'] = 1
        store.add_frame(FrameRecord(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            dataset=frame.dataset,
            scene_name=frame.scene_name,
            video_path=frame.video_path,
            frame_path=frame.frame_path,
            is_keyframe=frame.is_keyframe,
            metadata=metadata,
        ))
    candidate_detections = {
        candidate.frame_id: candidate.metadata.get('detections', {})
        for candidate in candidates
    }

    support_counts = {
        entity: sum(
            candidate.visibility.get(entity, 0.0) >= args.min_visibility
            for candidate in candidates
        )
        for entity in entities
    }
    print(
        f'[Intermediate] candidate visibility support for this pass: '
        f'{support_counts}'
    )
    insufficient = [
        entity for entity, count in support_counts.items()
        if count < args.min_support_frames
    ]
    if insufficient:
        print(
            '[Intermediate] This collection pass still lacks sufficient '
            'candidate support for: ' + ', '.join(insufficient)
        )

    effective_top_k = max(args.top_k, min(12, len(entities)))
    selected = select_question_keyframes(
        candidates=candidates,
        required_entities=entities,
        top_k=effective_top_k,
        min_temporal_gap=args.min_temporal_gap,
        min_joint_visibility=args.min_joint_visibility,
        min_support_frames=args.min_support_frames,
        intra_entity_min_gap=min(2.0, max(0.5, args.min_temporal_gap)),
    )
    if not selected:
        raise RuntimeError('No frames selected')
    selected = add_trajectory_bridge_frames(
        selected,
        candidates,
        max_total_frames=max(effective_top_k, 8),
        max_gap_seconds=15.0,
    )
    selected_ids = {item.frame_id for item in selected}
    print('selected frames:')
    for item in selected:
        print(f'  {item.frame_id} t={item.timestamp:.2f} score={item.score:.3f} {item.reason}')

    evidence_dir = scene_root / 'evidence'
    detection_dir = evidence_dir / 'detections'
    mask_dir = evidence_dir / 'masks'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    selection_path = evidence_dir / 'selected_frames.json'
    selection_path.write_text(json.dumps({
        'plan': plan,
        'selected': selected_to_dicts(selected),
        'candidates': [
            {
                'frame_id': candidate.frame_id,
                'timestamp': candidate.timestamp,
                'visibility': candidate.visibility,
                'sharpness': candidate.sharpness,
                'selected': candidate.frame_id in selected_ids,
                'detections': candidate.metadata.get('detections', {}),
            }
            for candidate in candidates
        ],
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    create_selection_contact_sheet(
        candidates, selected_ids, evidence_dir / 'selection_contact_sheet.jpg'
    )

    if args.selection_only:
        print(json.dumps({
            'question_id': args.question_id,
            'entities': entities,
            'selected_frame_ids': [item.frame_id for item in selected],
            'selection_json': str(selection_path),
            'selection_contact_sheet': str(evidence_dir / 'selection_contact_sheet.jpg'),
        }, ensure_ascii=False, indent=2))
        return

    store.clear_objects_and_observations()
    agent_memory_path_existing = evidence_dir / 'agent_memory.json'
    if args.round_index > 0 and agent_memory_path_existing.exists():
        agent_memory = AgentMemory.load(agent_memory_path_existing)
    else:
        agent_memory = AgentMemory()
    agent_memory.add('selected_frames', frame_ids=[item.frame_id for item in selected])
    observation_quality_by_entity = {entity: [] for entity in entities}
    selected_frames = [frames_by_id[item.frame_id] for item in selected]
    predictions, transform_info, extrinsic, intrinsic, vggt_image_tensor = run_question_vggt(
        selected_frames, device
    )

    geometry_dir = store.root_dir / 'geometry'
    geometry_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = geometry_dir / f'question_{args.question_id}_vggt_geometry.npz'
    np.savez_compressed(
        geometry_path,
        world_points=predictions['world_points'][0].float().cpu().numpy().astype(np.float16),
        world_points_conf=predictions['world_points_conf'][0].float().cpu().numpy().astype(np.float16),
        depth=predictions['depth'][0].float().cpu().numpy().astype(np.float16),
        depth_conf=predictions['depth_conf'][0].float().cpu().numpy().astype(np.float16),
        extrinsic=extrinsic.float().cpu().numpy(),
        intrinsic=intrinsic.float().cpu().numpy(),
        frame_ids=np.asarray([frame.frame_id for frame in selected_frames]),
        timestamps=np.asarray([frame.timestamp for frame in selected_frames], dtype=np.float32),
    )
    transform_path = geometry_dir / f'question_{args.question_id}_vggt_transform.json'
    transform_path.write_text(json.dumps(transform_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    extrinsic_cpu = extrinsic.detach().float().cpu()
    intrinsic_cpu = intrinsic.detach().float().cpu()
    if extrinsic_cpu.dim() == 4:
        extrinsic_cpu = extrinsic_cpu.squeeze(0)
    if intrinsic_cpu.dim() == 4:
        intrinsic_cpu = intrinsic_cpu.squeeze(0)
    for index, frame in enumerate(selected_frames):
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, :4] = extrinsic_cpu[index]
        store.add_camera_pose(CameraPose(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            t_world_cam=transform.tolist(),
            intrinsic=intrinsic_cpu[index].tolist(),
            confidence=None,
            metadata={'source': 'question_vggt', 'question_id': args.question_id},
        ))

    objects = {
        entity: MemoryObject(
            object_id=f'{entity}_00', category=entity, aliases=[entity],
            confidence=None, metadata={'question_id': args.question_id},
        )
        for entity in entities
    }
    frame_records = []
    object_points_accum = {entity: [] for entity in entities}
    object_colors_accum = {entity: [] for entity in entities}
    object_positions_accum = {entity: [] for entity in entities}

    # SAM2 is only needed for boxes that did not come with a mask (planner
    # overrides, or the grounding_dino path).  Build it lazily so a pure SAM3
    # run never pays for it.
    _sam2_holder: Dict[str, object] = {'predictor': None}

    def get_sam2_predictor():
        if _sam2_holder['predictor'] is None:
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=args.device)
            _sam2_holder['predictor'] = SAM2ImagePredictor(sam2_model)
        return _sam2_holder['predictor']

    for geometry_index, (selected_item, frame) in enumerate(zip(selected, selected_frames)):
        image = Image.open(frame.frame_path).convert('RGB')
        detections = candidate_detections.get(frame.frame_id, {})
        masks = {}
        if not detections:
            continue
        detection_image = detection_dir / f'{frame.frame_id}_detections.jpg'
        draw_detection_overlay(image, detections, detection_image)
        needs_sam2 = [
            entity for entity, detection in detections.items()
            if not (
                detection.get('mask_path')
                and Path(str(detection['mask_path'])).exists()
            )
        ]
        if needs_sam2:
            sam2_predictor = get_sam2_predictor()
            sam2_predictor.set_image(np.asarray(image))
        for entity, detection in detections.items():
            bbox = np.asarray(detection['bbox'], dtype=np.float32)
            cached_mask_path = detection.get('mask_path')
            if cached_mask_path and Path(str(cached_mask_path)).exists():
                # SAM3 produced the mask together with the box.
                mask = np.asarray(
                    Image.open(str(cached_mask_path)).convert('L')
                ) > 127
                sam_score = float(detection.get('score', 1.0))
            else:
                sam2_predictor = get_sam2_predictor()
                mask, mask_scores, _ = sam2_predictor.predict(
                    point_coords=None, point_labels=None,
                    box=bbox[None, :], multimask_output=False,
                )
                mask = np.asarray(mask[0], dtype=bool)
                sam_score = (
                    float(mask_scores[0]) if len(mask_scores)
                    else detection['score']
                )
            masks[entity] = mask

            # Enforce the task's mask bar *before* lifting to 3D, so a clipped
            # or fragmented mask never enters the point cloud.  Counting sets
            # no bar; size and distance do, because they measure the object.
            mask_stats = evaluate_mask_quality(mask)
            rejection = mask_rejection_reason(
                mask_stats,
                args.mask_boundary_max,
                args.mask_coverage_min,
                max_mask_border_sides=args.max_mask_border_sides,
            )
            if rejection is not None:
                mask_rejections.setdefault(entity, []).append({
                    'frame_id': frame.frame_id,
                    'reason': rejection,
                    'mask_boundary_ratio': mask_stats['mask_boundary_ratio'],
                    'mask_bbox_coverage': mask_stats['mask_bbox_coverage'],
                    'mask_touches_border': mask_stats['mask_touches_border'],
                    'mask_bbox_border_sides': mask_stats['mask_bbox_border_sides'],
                })
                print(
                    f'[Mask] {frame.frame_id} {entity}: rejected ({rejection}) '
                    f'boundary={mask_stats["mask_boundary_ratio"]:.2f} '
                    f'coverage={mask_stats["mask_bbox_coverage"]:.2f}',
                    flush=True,
                )
                continue

            mask_path = mask_dir / f'{frame.frame_id}_{entity}_mask.png'
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)

            vggt_mask = transform_mask_to_vggt(mask, transform_info[geometry_index])
            points = predictions['world_points'][0][geometry_index].float().cpu().numpy()[vggt_mask]
            confidence = predictions['world_points_conf'][0][geometry_index].float().cpu().numpy()[vggt_mask]
            vggt_image = (
                vggt_image_tensor[geometry_index]
                .permute(1, 2, 0)
                .numpy()
            )
            point_colors = (
                vggt_image[vggt_mask] * 255.0
            ).clip(0, 255).astype(np.uint8)
            finite = np.isfinite(points).all(axis=1)
            points, confidence, point_colors = (
                points[finite], confidence[finite], point_colors[finite]
            )
            if len(points) == 0:
                continue
            valid = confidence >= max(0.1, float(np.quantile(confidence, 0.5)))
            selected_points = points[valid] if valid.sum() >= 10 else points
            selected_colors = point_colors[valid] if valid.sum() >= 10 else point_colors
            position = np.median(selected_points, axis=0)
            object_points_accum[entity].append(selected_points)
            object_colors_accum[entity].append(selected_colors)
            object_positions_accum[entity].append(position.tolist())

            observation_id = f'q{args.question_id}_{frame.frame_id}_{entity}'
            observations_dir = geometry_dir / 'observations'
            observations_dir.mkdir(parents=True, exist_ok=True)
            observation_points_path = observations_dir / f'{observation_id}.npz'
            np.savez_compressed(
                observation_points_path,
                points=selected_points.astype(np.float16),
                colors=selected_colors.astype(np.uint8),
            )

            store.add_observation(Observation(
                observation_id=observation_id,
                object_id=objects[entity].object_id,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                dataset=args.dataset,
                scene_name=args.scene_name,
                video_path=frame.video_path,
                frame_path=frame.frame_path,
                bbox=detection['bbox'],
                mask_path=str(mask_path),
                position_3d=position.tolist(),
                confidence=min(detection['score'], sam_score),
                source=(
                    'question_sam3+vggt'
                    if detection.get('source') == 'sam3'
                    else 'question_dino+sam2+vggt'
                ),
                metadata={
                    'question_id': args.question_id,
                    'selection_score': selected_item.score,
                    'selection_reason': selected_item.reason,
                    'mask_score': sam_score,
                    'point_count': int(len(selected_points)),
                    # The extent of THIS view.  Track clustering needs it to
                    # decide whether two observations could be the same rigid
                    # object; without it every observation became its own
                    # instance and counting returned one per frame.
                    'point_extent': (
                        selected_points.max(axis=0) - selected_points.min(axis=0)
                    ).astype(float).tolist(),
                    'geometry_index': geometry_index,
                    'pointcloud_path': str(observation_points_path),
                },
            ))
            mask_quality = mask_stats
            observation_quality_by_entity[entity].append(ObservationQuality(
                entity=entity,
                frame_id=frame.frame_id,
                mask_area_ratio=mask_quality['mask_area_ratio'],
                mask_boundary_ratio=mask_quality['mask_boundary_ratio'],
                mask_bbox_coverage=mask_quality['mask_bbox_coverage'],
                mask_touches_border=bool(mask_quality.get('mask_touches_border')),
                mask_bbox_border_sides=int(
                    mask_quality.get('mask_bbox_border_sides') or 0
                ),
                point_count=int(len(selected_points)),
                confidence=min(detection['score'], sam_score),
                metadata={
                    'selection_score': selected_item.score,
                    'selection_reason': selected_item.reason,
                },
            ))
            agent_memory.add(
                'observation',
                entity=entity,
                frame_id=frame.frame_id,
                point_count=int(len(selected_points)),
                mask_quality=mask_quality,
            )

            objects[entity].first_seen = frame.timestamp if objects[entity].first_seen is None else min(objects[entity].first_seen, frame.timestamp)
            objects[entity].last_seen = frame.timestamp if objects[entity].last_seen is None else max(objects[entity].last_seen, frame.timestamp)
            objects[entity].confidence = min(detection['score'], sam_score) if objects[entity].confidence is None else max(objects[entity].confidence, min(detection['score'], sam_score))

        mask_image = evidence_dir / f'{frame.frame_id}_masks.jpg'
        draw_mask_overlay(image, masks, mask_image)
        frame_records.append({
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp,
            'detections': detections,
            'detection_image': str(detection_image),
            'mask_image': str(mask_image),
            'selection': selected_item,
        })

    for entity, chunks in object_points_accum.items():
        if not chunks:
            continue
        colors_chunks = object_colors_accum[entity]
        positions = object_positions_accum[entity]

        # An object's point cloud must come from views that agree on where the
        # object is.  Observations whose centroids sit far apart are either a
        # different instance or mis-registered by VGGT over a long time span;
        # concatenating them inflates every measurement.  Measured on a real
        # scene, three "stove" observations sat 0.33-0.63 m apart - as far apart
        # as the 0.6 m appliance itself - and their union measured 186 cm.
        selected_chunks = list(range(len(chunks)))
        cluster_report = {'clusters': 1, 'used': len(chunks), 'excluded': 0}
        if len(chunks) > 1 and len(positions) == len(chunks):
            probe = np.concatenate(chunks, axis=0)
            probe_extent = (probe.max(axis=0) - probe.min(axis=0)).astype(float)
            # Deliberately the DISTANCE rule here, not the extent rule: for
            # the stored cloud we want only mutually consistent views, so a
            # tight threshold is the point.  The extent rule is used for
            # deciding how many instances exist, where over-merging is the risk.
            eps = max(0.05, 0.2 * float(np.linalg.norm(probe_extent)))
            clusters = cluster_positions_by_distance(positions, eps)
            if len(clusters) > 1:
                # Rank by internal agreement first, data volume second.
                #
                # A cluster whose observations disagree about where the object
                # is stretches the union, so every measured extent comes out
                # too large: two "stove" views 0.18 m apart on a 0.6 m
                # appliance gave raw 80.6 / percentile 57.7 / obb 88.7 cm, all
                # three unreliable, while the single coherent view gave a
                # stable 48.5-60.6 cm.  When observations DO agree they tie on
                # agreement and the extra points win, so coverage is not lost.
                def agreement(indices):
                    if len(indices) < 2:
                        return 1.0
                    centroids = np.asarray([positions[i] for i in indices], float)
                    spread = float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0)))
                    joined = np.concatenate([chunks[i] for i in indices], axis=0)
                    extent = float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))
                    return 1.0 - spread / max(extent, 1e-6)

                def cluster_rank(indices):
                    points_total = sum(len(chunks[i]) for i in indices)
                    return (round(agreement(indices), 3), points_total)

                best = max(clusters, key=cluster_rank)
                selected_chunks = sorted(best)
                cluster_report = {
                    'clusters': len(clusters),
                    'used': len(selected_chunks),
                    'excluded': len(chunks) - len(selected_chunks),
                    'eps_m': eps,
                    'agreement': round(agreement(best), 3),
                }
                print(
                    f'[Cloud] {entity}: {len(clusters)} spatially separate '
                    f'observation cluster(s); using the most coherent one '
                    f'({len(selected_chunks)}/{len(chunks)} observations, '
                    f'agreement={agreement(best):.2f}, eps={eps:.3f}m).',
                    flush=True,
                )

        points = np.concatenate([chunks[i] for i in selected_chunks], axis=0)
        point_colors = np.concatenate(
            [colors_chunks[i] for i in selected_chunks], axis=0
        )
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        point_colors = point_colors[finite]
        np.savez_compressed(
            geometry_dir / f'{objects[entity].object_id}_points.npz',
            points=points.astype(np.float16),
            colors=point_colors.astype(np.uint8),
        )
        objects[entity].metadata['point_count'] = int(len(points))
        point_extent = (
            points.max(axis=0) - points.min(axis=0)
        ).astype(float).tolist()
        objects[entity].metadata['point_extent'] = point_extent
        objects[entity].metadata['cloud_clusters'] = cluster_report

        observations_with_position = [
            observation
            for observation in store.query_observations(
                object_id=objects[entity].object_id
            )
            if observation.position_3d is not None
        ]
        if observations_with_position:
            # Per-pair rule: see cluster_positions_by_extent.  A global eps
            # derived from the stored cloud's extent over-split the tracks
            # whenever that cloud happened to be one partial view.
            observation_extents = [
                (observation.metadata or {}).get('point_extent')
                or (observation.metadata or {}).get('extent')
                for observation in observations_with_position
            ]
            union_positions = np.asarray(
                [observation.position_3d for observation in observations_with_position],
                dtype=float,
            )
            span = union_positions.max(axis=0) - union_positions.min(axis=0)
            clusters = cluster_positions_by_extent(
                union_positions,
                observation_extents,
                fallback_eps=max(0.05, 0.2 * float(np.linalg.norm(span))),
            )
            print(
                f'[Tracks] {entity}: {len(clusters)} instance cluster(s) from '
                f'{len(observations_with_position)} observation(s).',
                flush=True,
            )
            for cluster_index, indices in enumerate(clusters):
                track_id = f'{entity}_{cluster_index:02d}'
                for observation_index in indices:
                    observation = observations_with_position[observation_index]
                    metadata = dict(observation.metadata or {})
                    metadata['track_id'] = track_id
                    observation.metadata = metadata
                    store.add_observation(observation)
            objects[entity].metadata['track_count'] = len(clusters)
            objects[entity].metadata['track_eps'] = eps
        store.upsert_object(objects[entity])

    rejected = {e: len(v) for e, v in mask_rejections.items() if v}
    if rejected:
        print(
            f'[Mask] rejected {sum(rejected.values())} observation(s) that '
            f'failed the {mask_purpose} bar: {rejected}',
            flush=True,
        )

    checker = EvidenceSufficiencyChecker(
        min_observations_per_entity=args.min_support_frames,
        min_points_per_entity=1000,
        # Same bar as the collection filter, taken from the task constraint.
        # Otherwise counting - which declares no bar - would still be rejected
        # here by the checker's built-in shape defaults.
        max_boundary_ratio=(
            args.mask_boundary_max if args.mask_boundary_max is not None else 1.0
        ),
        min_bbox_coverage=(
            args.mask_coverage_min if args.mask_coverage_min is not None else 0.0
        ),
        max_mask_border_sides=(
            args.max_mask_border_sides
            if args.max_mask_border_sides is not None else 4
        ),
    )
    evidence_status = checker.check(
        required_entities=entities,
        observations_by_entity=observation_quality_by_entity,
        required_evidence=plan.get('required_evidence', []),
    )
    agent_memory.add('evidence_status', status=evidence_status.to_dict())

    next_evidence_request = None
    if not evidence_status.sufficient:
        requests = []
        for entity, missing in evidence_status.missing.items():
            if entity == '_global':
                if 'metric_scale' in missing:
                    requests.append({
                        'tool_name': 'metric_scale_estimation',
                        'entity': None,
                        'reason': 'missing_metric_scale',
                    })
                continue
            entity_observations = observation_quality_by_entity.get(entity, [])
            anchor_ids = [obs.frame_id for obs in entity_observations]
            anchor_timestamps = [
                frames_by_id[frame_id].timestamp
                for frame_id in anchor_ids
                if frame_id in frames_by_id
                and frames_by_id[frame_id].timestamp is not None
            ]
            neighbor_timestamps = []
            for anchor_timestamp in anchor_timestamps:
                neighbor_timestamps.extend(build_neighbor_timestamps(
                    [anchor_timestamp],
                    offsets_seconds=(-1.0, -0.5, 0.5, 1.0),
                    min_timestamp=0.0,
                    max_timestamp=max(
                        frame.timestamp for frame in all_frames
                        if frame.timestamp is not None
                    ),
                ))
            neighbor_timestamps = sorted(set(neighbor_timestamps))
            video_path = next(
                frame.video_path for frame in all_frames if frame.video_path
            )
            extracted = extract_video_frames_at_timestamps(
                video_path,
                scene_root / 'frames' / 'neighbors',
                neighbor_timestamps,
            )
            for item in extracted:
                store.add_frame(FrameRecord(
                    frame_id=item['frame_id'],
                    timestamp=item['timestamp'],
                    dataset=args.dataset,
                    scene_name=args.scene_name,
                    video_path=video_path,
                    frame_path=item['frame_path'],
                    is_keyframe=False,
                    metadata={
                        'generated_by': 'temporal_neighbor_search',
                        'role': EVIDENCE_FRAME_ROLE,
                        'entity': entity,
                        'anchor_frame_ids': anchor_ids,
                    },
                ))
            requests.append({
                'tool_name': 'temporal_neighbor_search',
                'entity': entity,
                'anchor_frame_ids': anchor_ids,
                'candidate_frame_ids': [item['frame_id'] for item in extracted],
                'candidate_frame_paths': [item['frame_path'] for item in extracted],
                'reason': missing,
            })
        next_evidence_request = {'requests': requests}
        agent_memory.add('next_evidence_request', request=next_evidence_request)

    evidence_status_path = evidence_dir / 'evidence_status.json'
    evidence_status_path.write_text(
        json.dumps(evidence_status.to_dict(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    agent_memory_path = evidence_dir / 'agent_memory.json'
    agent_memory.save(agent_memory_path)
    if next_evidence_request is not None:
        (evidence_dir / 'next_evidence_request.json').write_text(
            json.dumps(next_evidence_request, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    object_points = {}
    object_colors = {}
    for entity in entities:
        path = geometry_dir / f'{objects[entity].object_id}_points.npz'
        if path.exists():
            data = np.load(path)
            object_points[entity] = data['points'].astype(np.float32)
            if 'colors' in data:
                object_colors[entity] = data['colors'].astype(np.uint8)
    plot_object_points_gca(
        object_points,
        object_colors=object_colors,
        output_path=evidence_dir / 'object_pointclouds.png',
    )
    try:
        scene_image = visualize_3d_scene(
            points=predictions['world_points'][0].detach().float().cpu(),
            points_conf=predictions['world_points_conf'][0].detach().float().cpu(),
            image_tensor=vggt_image_tensor,
            camera_extrinsics=extrinsic_cpu,
            conf_thres=10.0,
            show_cam=True,
        )
        scene_image.save(evidence_dir / 'scene_reconstruction.png')
    except Exception as exc:
        print(f'[Warning] GCA scene visualization failed: {exc}', flush=True)
    plot_camera_trajectory(selected_frames, extrinsic, evidence_dir / 'camera_trajectory.png')

    report_rows = []
    for record in frame_records:
        rel_detection = Path(record['detection_image']).relative_to(evidence_dir)
        rel_mask = Path(record['mask_image']).relative_to(evidence_dir)
        report_rows.append(f"""
        <section><h2>Frame {record['frame_id']} | t={record['timestamp']:.2f}s</h2>
        <p>selection: {record['selection'].reason}, score={record['selection'].score:.3f}</p>
        <div style="display:flex;gap:16px;flex-wrap:wrap">
        <figure><img src="{rel_detection}" width="420"><figcaption>GroundingDINO detection</figcaption></figure>
        <figure><img src="{rel_mask}" width="420"><figcaption>SAM2 mask</figcaption></figure>
        </div><pre>{json.dumps(record['detections'], ensure_ascii=False, indent=2)}</pre></section>
        """)
    report_path = evidence_dir / 'index.html'
    report_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>VSI Evidence Report</title>
<style>body{{font-family:sans-serif;margin:24px}}img{{border:1px solid #ccc}}section{{margin-bottom:32px;border-bottom:1px solid #ddd;padding-bottom:20px}}pre{{background:#f6f8fa;padding:12px;overflow-x:auto}}</style>
</head><body><h1>VSI-Bench Targeted Evidence Report</h1>
<pre>{json.dumps(plan, ensure_ascii=False, indent=2)}</pre>
<h2>Frame Selection</h2><img src="selection_contact_sheet.jpg" width="1000">
<h2>3D Object Points</h2><img src="object_pointclouds.png" width="600">
<h2>GCA Scene Reconstruction</h2><img src="scene_reconstruction.png" width="900">
<h2>Camera Trajectory</h2><img src="camera_trajectory.png" width="600">
{''.join(report_rows)}</body></html>""", encoding='utf-8')

    scene_path = store.root_dir / 'scene.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    scene.setdefault('question_evidence', {})[str(args.question_id)] = {
        'plan_path': str(plan_path),
        'selected_frames': selected_to_dicts(selected),
        'geometry_path': str(geometry_path),
        'transform_path': str(transform_path),
        'report_path': str(report_path),
    }
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    store.export_parquet()

    entity_missing = {
        entity: missing
        for entity, missing in evidence_status.missing.items()
        if entity != '_global'
    }
    if (
        args.auto_expand
        and entity_missing
        and args.round_index + 1 < args.max_rounds
    ):
        temporal_requests = [
            request for request in (next_evidence_request or {}).get('requests', [])
            if request.get('tool_name') == 'temporal_neighbor_search'
            and request.get('candidate_frame_ids')
        ]
        if temporal_requests:
            forward_args = []
            skip_next = False
            for index, value in enumerate(sys.argv[1:]):
                if skip_next:
                    skip_next = False
                    continue
                if value == '--round-index':
                    skip_next = True
                    continue
                if value == '--max-rounds':
                    skip_next = True
                    continue
                if value.startswith('--round-index='):
                    continue
                if value.startswith('--max-rounds='):
                    continue
                forward_args.append(value)
            command = [
                sys.executable,
                '-m',
                'entrypoints.collect_vsibench_evidence',
                *forward_args,
                '--round-index',
                str(args.round_index + 1),
                '--max-rounds',
                str(args.max_rounds),
            ]
            print('Launching next evidence round:', ' '.join(command), flush=True)
            os.execv(sys.executable, command)

    summary = store.summary()
    summary.update({
        'question_id': args.question_id,
        'entities': entities,
        'selected_frame_ids': [item.frame_id for item in selected],
        'observations_created': sum(len(item) for item in object_points_accum.values()),
        'report': str(report_path),
        'selection_contact_sheet': str(evidence_dir / 'selection_contact_sheet.jpg'),
        'evidence_sufficient': evidence_status.sufficient,
        'evidence_status': evidence_status.to_dict(),
        'next_evidence_request': next_evidence_request,
        'agent_memory': str(agent_memory_path),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
