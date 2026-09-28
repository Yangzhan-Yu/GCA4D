import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image

from openai import OpenAI

from tools.apis.four_d_memory import open_vsibench_memory
from tools.apis.api_budget import reserve_api_call
from tools.apis.temporal_neighbor_search import extract_video_frames_at_timestamps
from workflow.utils.parse_utils import parse_json_str


ALLOWED_ENTRYPOINTS = {
    'entrypoints.collect_vsibench_evidence',
    'entrypoints.estimate_vsibench_metric_scale',
    'entrypoints.estimate_vsibench_object_size',
}


def get_scene_root(context: Dict[str, Any]) -> Path:
    return Path(context['scene_root'])


def open_memory(context: Dict[str, Any]):
    return open_vsibench_memory(
        data_root=context['data_root'],
        dataset=context['dataset'],
        scene_name=context['scene_name'],
        scene_root=context['scene_root'],
    )


def get_question_evidence(context: Dict[str, Any], question_id: int) -> Dict[str, Any]:
    store = open_memory(context)
    scene_path = store.root_dir / 'scene.json'
    if not scene_path.exists():
        return {}
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    return scene.get('question_evidence', {}).get(str(question_id), {})


def run_entrypoint(
    context: Dict[str, Any],
    module: str,
    args: List[str],
    timeout: int = 600,
) -> Dict[str, Any]:
    if module not in ALLOWED_ENTRYPOINTS:
        return {'error': f'Entrypoint not allowed: {module}'}
    command = [sys.executable, '-m', module, *[str(value) for value in args]]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        'returncode': completed.returncode,
        'output': completed.stdout[-20000:],
    }


def list_memory_frames(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    store = open_memory(context)
    return [
        {
            'frame_id': frame.frame_id,
            'timestamp': frame.timestamp,
            'frame_path': frame.frame_path,
            'video_path': frame.video_path,
            'metadata': frame.metadata or {},
        }
        for frame in store.query_frames()
    ]


def list_memory_frame_paths(context: Dict[str, Any]) -> List[str]:
    return [
        frame['frame_path']
        for frame in list_memory_frames(context)
        if frame.get('frame_path')
    ]


def sample_video_frames(
    context: Dict[str, Any],
    max_frames: int = 12,
    subdir: str = 'generated_frames',
    stride_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    video_path = str(
        Path(context['data_root']) / 'videos' / context['dataset'] /
        f"{context['scene_name']}.mp4"
    )
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f'Invalid video metadata: {video_path}')
    duration = frame_count / fps
    if stride_seconds is None:
        stride_seconds = duration / max(1, max_frames - 1)
    timestamps = [
        index * float(stride_seconds)
        for index in range(max_frames)
        if index * float(stride_seconds) <= duration
    ]
    frame_dir = get_scene_root(context) / 'frames' / subdir
    return extract_video_frames_at_timestamps(
        video_path=video_path,
        frame_dir=frame_dir,
        timestamps=timestamps,
    )


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def image_content(image_or_path):
    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path).convert('RGB')
    else:
        image = image_or_path.convert('RGB')
    return {
        'type': 'image_url',
        'image_url': {'url': image_to_data_uri(image)},
    }


def vlm_chat(
    context: Dict[str, Any],
    content: List[Dict[str, Any]],
    max_tokens: int = 2048,
) -> str:
    reserve_api_call('generated_tool_vlm_chat')
    client = OpenAI(
        base_url=context.get('base_url') or os.environ['AGENT_COT_REASONER_BASE_URL'],
        api_key=context.get('api_key') or os.environ['AGENT_COT_REASONER_API_KEY'],
        timeout=120.0,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=context['model'],
        messages=[{'role': 'user', 'content': content}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.95,
    )
    return response.choices[0].message.content


def parse_json_object(content: str) -> Dict[str, Any]:
    result, _ = parse_json_str(content)
    return result


def save_json(context: Dict[str, Any], name: str, value: Any) -> str:
    output_dir = get_scene_root(context) / 'evidence' / 'generated'
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return str(path)


__all__ = [
    'get_scene_root',
    'image_content',
    'get_question_evidence',
    'image_to_data_uri',
    'list_memory_frames',
    'list_memory_frame_paths',
    'open_memory',
    'parse_json_object',
    'sample_video_frames',
    'save_json',
    'run_entrypoint',
    'vlm_chat',
]
