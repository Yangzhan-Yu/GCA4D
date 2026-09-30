"""Batch SAM3 detection + segmentation worker.

SAM3 needs torch>=2.7 while the rest of the pipeline runs in the gca env
(torch 2.5), so SAM3 is invoked as a subprocess with ``$PYTHON_SAM3``.  This
worker loads the checkpoint once and then processes every requested frame, so
the ~19s weight load is paid once per evidence-collection pass instead of once
per frame.

Input (``--requests``)::

    {
      "prompts": ["chair", "sofa"],
      "frames": [{"frame_id": "001800", "image_path": "/abs/path.jpg"}, ...]
    }

Output (``--output-json``)::

    {
      "prompts": [...],
      "model_loaded_seconds": 18.8,
      "inference_seconds": 3.2,
      "results": {
        "001800": {
          "chair": [{"score": 0.91, "bbox": [x1,y1,x2,y2],
                     "mask_area_ratio": 0.08, "mask_path": "/abs/mask.png"}],
          ...
        }
      },
      "errors": {"001900": "..."}
    }

Only the top-scoring mask per prompt is exported by default (``--max-per-prompt``);
raise it to keep alternatives for Planner disambiguation.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_mask_quality():
    """Load the metric helper by path (this env has no gca stack)."""
    import importlib.util

    path = REPO_ROOT / 'tools' / 'utils' / 'mask_metrics.py'
    spec = importlib.util.spec_from_file_location('_gca_mask_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mask_quality


def load_segmenter_class():
    """Import the adapter without initialising the whole ``tools`` package."""
    import importlib.util

    module_path = REPO_ROOT / 'tools' / 'apis' / 'sam3_local.py'
    spec = importlib.util.spec_from_file_location('_gca_sam3_local', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sam3TextSegmenter


def parse_args():
    parser = argparse.ArgumentParser('Batch SAM3 text-prompt segmentation')
    parser.add_argument('--requests', required=True,
                        help='JSON file with prompts + frame list')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--mask-dir', default=None,
                        help='Directory for mask PNGs (default <output dir>/masks)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--resolution', type=int, default=1008)
    parser.add_argument('--max-per-prompt', type=int, default=1,
                        help='Keep the N highest-scoring masks per prompt')
    parser.add_argument('--no-autocast', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    from PIL import Image

    requests = json.loads(Path(args.requests).read_text(encoding='utf-8'))
    prompts: List[str] = list(requests.get('prompts') or [])
    frames: List[Dict[str, Any]] = list(requests.get('frames') or [])
    if not prompts:
        raise ValueError('requests.prompts must not be empty')

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else output_path.parent / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'[SAM3] {len(frames)} frame(s), {len(prompts)} prompt(s), '
        f'device={args.device}, confidence={args.confidence}',
        flush=True,
    )

    mask_quality = load_mask_quality()
    segmenter = load_segmenter_class()(
        device=args.device,
        confidence_threshold=args.confidence,
        resolution=args.resolution,
        use_bf16_autocast=not args.no_autocast,
    )

    started = time.perf_counter()
    segmenter._ensure_loaded()
    load_seconds = time.perf_counter() - started
    print(f'[SAM3] model loaded in {load_seconds:.1f}s', flush=True)

    results: Dict[str, Dict[str, List[Dict]]] = {}
    errors: Dict[str, str] = {}
    started = time.perf_counter()
    for index, frame in enumerate(frames, start=1):
        frame_id = str(frame.get('frame_id'))
        image_path = frame.get('image_path')
        if not image_path or not Path(image_path).exists():
            errors[frame_id] = f'image not found: {image_path}'
            continue
        try:
            image = Image.open(image_path).convert('RGB')
            outcome = segmenter.segment(image, prompts)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            errors[frame_id] = f'{type(exc).__name__}: {exc}'
            print(f'[SAM3] frame {frame_id} FAILED: {errors[frame_id]}', flush=True)
            continue

        frame_result: Dict[str, List[Dict]] = {}
        for prompt in prompts:
            hits = sorted(outcome.get(prompt, []), key=lambda d: -d['score'])
            hits = hits[: max(1, args.max_per_prompt)]
            exported = []
            for rank, hit in enumerate(hits):
                mask = np.asarray(hit['mask'], dtype=bool)
                mask_name = f'{frame_id}_{prompt}_{rank}.png'
                mask_path = mask_dir / mask_name
                Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
                entry = {
                    'score': float(hit['score']),
                    'bbox': [float(v) for v in hit['bbox']],
                    'mask_area_ratio': float(hit['mask_area_ratio']),
                    'mask_path': str(mask_path),
                    'detector': 'sam3',
                }
                # Record the shape statistics: a later question that measures
                # the object instead of just counting it needs to know whether
                # this mask is truncated or fragmented.
                entry.update(mask_quality(mask))
                exported.append(entry)
            frame_result[prompt] = exported
        results[frame_id] = frame_result
        if index % 10 == 0 or index == len(frames):
            print(f'[SAM3] {index}/{len(frames)} frames', flush=True)
    inference_seconds = time.perf_counter() - started

    payload = {
        'prompts': prompts,
        'model_loaded_seconds': load_seconds,
        'inference_seconds': inference_seconds,
        'results': results,
        'errors': errors,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    total = sum(
        len(hits) for frame_result in results.values()
        for hits in frame_result.values()
    )
    per_frame = inference_seconds / max(1, len(frames))
    print(
        f'[SAM3] done: {total} mask(s) over {len(results)} frame(s) in '
        f'{inference_seconds:.2f}s ({per_frame:.3f}s/frame)',
        flush=True,
    )
    if errors:
        print(f'[SAM3] {len(errors)} frame(s) failed', flush=True)
    print(f'[SAM3] wrote {output_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
