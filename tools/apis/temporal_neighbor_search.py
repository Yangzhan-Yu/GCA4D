from typing import Iterable, List, Optional, Sequence


def find_temporal_neighbors(
    frames: Sequence,
    anchor_frame_ids: Iterable[str],
    radius_seconds: float = 2.0,
    max_new_frames: int = 4,
    exclude_frame_ids: Optional[Iterable[str]] = None,
) -> List:
    exclude = set(exclude_frame_ids or [])
    anchors = [
        frame for frame in frames
        if frame.frame_id in set(anchor_frame_ids)
    ]
    candidates = []
    for frame in frames:
        if frame.frame_id in exclude:
            continue
        if frame.timestamp is None:
            continue
        distance = min(
            (abs(frame.timestamp - anchor.timestamp)
             for anchor in anchors if anchor.timestamp is not None),
            default=float('inf'),
        )
        if distance <= radius_seconds and distance > 0:
            candidates.append((distance, frame.timestamp, frame))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = []
    seen = set()
    for _, _, frame in candidates:
        if frame.frame_id in seen:
            continue
        selected.append(frame)
        seen.add(frame.frame_id)
        if len(selected) >= max_new_frames:
            break
    return selected


__all__ = [
    'find_temporal_neighbors',
    'build_neighbor_timestamps',
    'extract_video_frames_at_timestamps',
]


def build_neighbor_timestamps(
    anchor_timestamps,
    offsets_seconds=(-1.0, -0.5, 0.5, 1.0),
    min_timestamp: float = 0.0,
    max_timestamp: float | None = None,
):
    timestamps = []
    for anchor in anchor_timestamps:
        for offset in offsets_seconds:
            timestamp = float(anchor) + float(offset)
            if timestamp < min_timestamp:
                continue
            if max_timestamp is not None and timestamp > max_timestamp:
                continue
            timestamps.append(timestamp)
    return sorted(dict.fromkeys(round(value, 6) for value in timestamps))


def extract_video_frames_at_timestamps(
    video_path,
    frame_dir,
    timestamps,
):
    import os
    from pathlib import Path

    import cv2

    video_path = str(video_path)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f'Invalid video metadata: {video_path}')

        extracted = []
        for timestamp in sorted(set(float(value) for value in timestamps)):
            frame_index = min(
                frame_count - 1,
                max(0, int(round(timestamp * fps))),
            )
            frame_path = frame_dir / f'frame_{frame_index:06d}.jpg'
            if not frame_path.exists():
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                if not cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                ):
                    continue
            extracted.append({
                'frame_id': f'{frame_index:06d}',
                'timestamp': frame_index / fps,
                'frame_path': str(frame_path),
            })
    finally:
        capture.release()

    return extracted
