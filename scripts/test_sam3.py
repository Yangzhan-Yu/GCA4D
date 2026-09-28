#!/usr/bin/env python
"""Smoke-test SAM3 text-prompted detection + segmentation on one frame.

Verifies that the SAM3 checkpoint loads and that a text prompt returns usable
boxes and masks *before* it gets wired into the evidence pipeline.

    source scripts/gca_env.sh
    python scripts/test_sam3.py \
        --video data/vsibench/videos/arkitscenes/41069025.mp4 \
        --timestamp 60 \
        --prompts chair sofa stove

Writes an overlay that keeps the original image colours (mask blended at 45%)
plus a JSON summary, so weak or wrong detections are obvious by eye.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Distinct colours per instance, cycled as needed.
PALETTE = [
    (230, 60, 60), (60, 200, 90), (70, 120, 240), (240, 180, 40),
    (200, 70, 220), (40, 200, 210), (250, 120, 170), (150, 150, 60),
]


def parse_args():
    parser = argparse.ArgumentParser('SAM3 single-frame smoke test')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='Path to an image file')
    source.add_argument('--video', help='Path to a video file')
    parser.add_argument('--timestamp', type=float, default=0.0,
                        help='Seconds into --video to sample')
    parser.add_argument('--prompts', nargs='+', required=True,
                        help='Text prompts, e.g. chair sofa')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--resolution', type=int, default=1008,
                        help='SAM3 processor resolution (default 1008)')
    parser.add_argument('--output-dir', default='work_dir/sam3_smoke')
    return parser.parse_args()


def load_frame(args):
    from PIL import Image

    if args.image:
        path = Path(args.image)
        if not path.exists():
            raise FileNotFoundError(f'Image not found: {path}')
        return Image.open(path).convert('RGB'), path.stem

    import cv2

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f'Video not found: {video}')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {video}')
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, args.timestamp) * 1000.0)
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Could not read frame at {args.timestamp}s')
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    stem = f'{video.stem}_{args.timestamp:07.2f}s'.replace('.', '_')
    return Image.fromarray(frame), stem


def draw_overlay(image, detections):
    """Blend each mask in its own colour, preserving the original image."""
    import cv2

    canvas = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    height, width = canvas.shape[:2]
    for index, detection in enumerate(detections):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        mask = np.asarray(detection['mask'], dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        canvas[mask] = 0.55 * canvas[mask] + 0.45 * color
    canvas = canvas.clip(0, 255).astype(np.uint8)

    for index, detection in enumerate(detections):
        color = PALETTE[index % len(PALETTE)]
        mask = np.asarray(detection['mask'], dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, color, 3)
        x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{index}:{detection['label']} "
            f"{detection['score']:.2f} area={detection['mask_area_ratio']:.3f}"
        )
        cv2.putText(
            canvas, label, (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, (x1 + 4, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()

    from PIL import Image

    from tools.apis.sam3_local import Sam3TextSegmenter

    checkpoint = os.environ.get('SAM3_CHECKPOINT', '<unset>')
    print(f'[SAM3] checkpoint : {checkpoint}')
    print(f'[SAM3] exists     : {Path(checkpoint).exists() if checkpoint != "<unset>" else False}')
    print(f'[SAM3] device     : {args.device}')
    print(f'[SAM3] prompts    : {args.prompts}')

    image, stem = load_frame(args)
    print(f'[SAM3] frame      : {stem} ({image.width}x{image.height})')

    segmenter = Sam3TextSegmenter(
        device=args.device,
        confidence_threshold=args.confidence,
        resolution=args.resolution,
    )

    started = time.perf_counter()
    try:
        segmenter._ensure_loaded()
    except Exception as exc:  # noqa: BLE001
        print(f'[SAM3] LOAD FAILED: {type(exc).__name__}: {exc}')
        return 1
    load_seconds = time.perf_counter() - started
    print(f'[SAM3] model loaded in {load_seconds:.1f}s (weights loaded once)')

    started = time.perf_counter()
    try:
        results = segmenter.segment(image, args.prompts)
    except Exception as exc:  # noqa: BLE001
        print(f'[SAM3] INFERENCE FAILED: {type(exc).__name__}: {exc}')
        return 1
    infer_seconds = time.perf_counter() - started

    flat = []
    summary = {}
    for prompt in args.prompts:
        hits = sorted(results.get(prompt, []), key=lambda d: -d['score'])
        summary[prompt] = [
            {
                'score': round(hit['score'], 4),
                'bbox': [round(float(v), 1) for v in hit['bbox']],
                'mask_area_ratio': round(hit['mask_area_ratio'], 4),
            }
            for hit in hits
        ]
        flat.extend(hits)
        print(f'  {prompt:16s} -> {len(hits)} mask(s)')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(image, flat)
    overlay_path = output_dir / f'{stem}_sam3.jpg'
    Image.fromarray(overlay).save(overlay_path, quality=92)
    json_path = output_dir / f'{stem}_sam3.json'
    json_path.write_text(
        json.dumps(
            {'image': stem, 'prompts': args.prompts, 'results': summary},
            ensure_ascii=False, indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    total = sum(len(v) for v in summary.values())
    print(f'\n[SAM3] {total} detection(s) in {infer_seconds:.2f}s '
          f'({infer_seconds / max(1, len(args.prompts)):.2f}s per prompt)')
    print(f'[SAM3] overlay      : {overlay_path}')
    print(f'[SAM3] summary json : {json_path}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if total == 0:
        print(
            '\nNo detections. Try --confidence 0.3, different --timestamp, or '
            'check that box scores are not being filtered too aggressively.'
        )
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
