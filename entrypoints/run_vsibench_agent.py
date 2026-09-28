import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.apis.agent_memory import AgentMemory
from tools.apis.api_budget import (
    ApiBudgetExceeded,
    configure_api_budget,
    get_api_budget,
    reserve_api_call,
)
from tools.apis.vlm_grounding import image_to_data_uri
from tools.apis.grounding_dino_local import GroundingDinoDetector
from tools.apis.llm_endpoint import (
    async_chat_text,
    create_async_client,
    create_sync_client,
    resolve_endpoint,
)
from tools.apis.four_d_memory import (
    CONTEXT_FRAME_ROLE,
    EVIDENCE_FRAME_ROLE,
    FrameRecord,
    VISIBILITY_SCAN_FRAME_ROLE,
    is_evidence_candidate_frame,
    open_vsibench_memory,
)
from tools.apis.temporal_neighbor_search import (
    build_neighbor_timestamps,
    extract_video_frames_at_timestamps,
)
from workflow.agentic.planner_loop import PlannerLoop
from workflow.constraints.pipeline import (
    build_context as build_constraint_context,
    load_constraint,
    load_result_store,
    run_operation as run_constraint_operation,
    save_constraint,
)
from workflow.constraints.runtime import (
    load_bindings,
    merge_bindings,
    resolve_extra_inputs,
    resolve_points,
    save_bindings,
)
from workflow.constraints.task_constraints import compile_task_constraint
from workflow.constraints.validator import validate_operation as validate_op
from workflow.constraints.validator import validate_result as validate_res
from workflow.agentic.tool_synthesizer import (
    ToolSynthesizer,
    load_generated_tool_specs,
)
from workflow.nodes.evidence_planner import QuestionEvidencePlanner
from workflow.agentic.tool_registry import ToolRegistry, ToolSpec
from workflow.utils.parse_utils import parse_json_str


def parse_args():
    parser = argparse.ArgumentParser('Run agentic VSI-Bench evidence loop')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--question-id', type=int, required=True)
    parser.add_argument('--data-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench')
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=5,
        help=(
            'Safety cap on complete evidence-gathering rounds. Each round may '
            'contain multiple tool calls.'
        ),
    )
    parser.add_argument(
        '--max-steps-per-round',
        type=int,
        default=None,
        help=(
            'Optional safety cap on tool/decision steps inside one evidence '
            'round. By default the Agent decides when the round ends.'
        ),
    )
    parser.add_argument(
        '--max-total-steps',
        type=int,
        default=25,
        help='Safety cap on total tool/decision steps across all rounds.',
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Deprecated alias for --max-steps-per-round.',
    )
    parser.add_argument('--max-tool-failures', type=int, default=2)
    parser.add_argument(
        '--max-api-calls',
        type=int,
        default=120,
        help='Maximum number of Qwen API calls for this run (shared by all subprocesses).',
    )
    parser.add_argument(
        '--allow-tool-generation',
        action='store_true',
        help=(
            'Allow the Planner to synthesise/repair and persist new tools. '
            'Disabled by default so the core constraint experiment uses a '
            'fixed tool set and stays comparable across questions.'
        ),
    )
    parser.add_argument('--force-plan', action='store_true')
    parser.add_argument('--reset-agent-memory', action='store_true')
    return parser.parse_args()


def run_subprocess(command, cwd):
    print(f'[Subprocess] Starting: {" ".join(command)}', flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip('\n')
        lines.append(line)
        print(f'[ToolLog] {line}', flush=True)
    returncode = process.wait()
    print(
        f'[Subprocess] Finished with returncode={returncode}',
        flush=True,
    )
    return {
        'returncode': returncode,
        'output': '\n'.join(lines[-400:])[-20000:],
    }


def load_plan(scene_root: Path, question_id: int):
    path = scene_root / 'question_plans' / f'{question_id}.json'
    if not path.exists():
        raise FileNotFoundError(f'Question plan not found: {path}')
    return path, json.loads(path.read_text(encoding='utf-8'))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def register_tools(registry: ToolRegistry):
    def scan_entity_visibility(context: Dict[str, Any], **args):
        entities = [str(item).strip().lower() for item in args.get('entities', [])]
        if not entities:
            return {'error': 'entities must not be empty'}
        stride = float(args.get('scan_stride_seconds', 4.0))
        max_frames = int(args.get('max_scan_frames', 80))
        store = context['store']
        video_path = str(
            context['data_root'] / 'videos' / context['dataset'] /
            f"{context['scene_name']}.mp4"
        )
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return {'error': f'Failed to open video: {video_path}'}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frame_count <= 0:
            return {'error': f'Invalid video metadata: {video_path}'}
        max_timestamp = frame_count / fps
        timestamps = [index * stride for index in range(max_frames) if index * stride <= max_timestamp]
        extracted = extract_video_frames_at_timestamps(
            video_path=video_path,
            frame_dir=context['scene_root'] / 'frames' / 'visibility_scan',
            timestamps=timestamps,
        )
        visible_frames_by_entity = {entity: [] for entity in entities}

        def detection_quality(detection, image):
            x1, y1, x2, y2 = detection['bbox']
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_fraction = area / max(1.0, float(image.width * image.height))
            confidence = max(0.0, min(1.0, float(detection.get('score', 1.0))))
            return confidence * min(1.0, (area_fraction / 0.08) ** 0.5)

        for item in extracted:
            image = Image.open(item['frame_path']).convert('RGB')
            detector = context.get('grounding_dino_detector')
            if detector is None:
                detector = GroundingDinoDetector(device=context['device'])
                context['grounding_dino_detector'] = detector
            detections = detector.detect(image, entities)
            frame_visibility = {entity: 0.0 for entity in entities}
            for entity in entities:
                detection = detections.get(entity)
                if detection is None:
                    continue
                quality = detection_quality(detection, image)
                frame_visibility[entity] = quality
                visible_frames_by_entity[entity].append({
                    'frame_id': item['frame_id'],
                    'timestamp': item['timestamp'],
                    'bbox': detection['bbox'],
                    'quality': quality,
                    'confidence': float(detection.get('score', 0.0)),
                })
            store.add_frame(FrameRecord(
                frame_id=item['frame_id'],
                timestamp=item['timestamp'],
                dataset=context['dataset'],
                scene_name=context['scene_name'],
                video_path=video_path,
                frame_path=item['frame_path'],
                is_keyframe=False,
                metadata={
                    'generated_by': 'scan_entity_visibility',
                    'role': VISIBILITY_SCAN_FRAME_ROLE,
                    'visibility': frame_visibility,
                    'detections': detections,
                },
            ))

        result = {}
        for entity, visible_frames in visible_frames_by_entity.items():
            visible_times = [record['timestamp'] for record in visible_frames]
            intervals = []
            if visible_times:
                start = previous = visible_times[0]
                for time in visible_times[1:]:
                    if time - previous > stride * 1.5:
                        intervals.append([start, previous])
                        start = time
                    previous = time
                intervals.append([start, previous])
            best = (
                max(visible_frames, key=lambda record: record['quality'])
                if visible_frames else None
            )
            result[entity] = {
                'first_seen': min(visible_times) if visible_times else None,
                'last_seen': max(visible_times) if visible_times else None,
                'intervals': intervals,
                'anchor_frame_id': best['frame_id'] if best else None,
                'anchor_timestamp': best['timestamp'] if best else None,
                'anchor_quality': best['quality'] if best else None,
                'visible_frames': visible_frames,
            }

        anchor_frames = [
            {
                'entity': entity,
                'frame_id': payload['anchor_frame_id'],
                'timestamp': payload['anchor_timestamp'],
            }
            for entity, payload in result.items()
            if payload['anchor_frame_id'] is not None
        ]
        suggested_bridge_window = None
        if len(anchor_frames) >= 2:
            suggested_bridge_window = {
                'start_time': min(item['timestamp'] for item in anchor_frames),
                'end_time': max(item['timestamp'] for item in anchor_frames),
                'anchor_frame_ids': [
                    item['frame_id'] for item in anchor_frames
                ],
                'anchors': anchor_frames,
            }
        return {
            'visibility': result,
            'scanned_frames': len(extracted),
            'suggested_bridge_window': suggested_bridge_window,
        }

    def find_bridge_frames(context: Dict[str, Any], **args):
        start_time = float(args['start_time'])
        end_time = float(args['end_time'])
        interval = float(args.get('interval_seconds', 3.0))
        padding = float(args.get('padding_seconds', 1.0))
        anchor_frame_ids = [
            str(value) for value in args.get('anchor_frame_ids', [])
        ]
        if end_time <= start_time:
            return {'error': 'end_time must be greater than start_time'}
        store = context['store']
        video_path = str(
            context['data_root'] / 'videos' / context['dataset'] /
            f"{context['scene_name']}.mp4"
        )
        frames_by_id = {
            frame.frame_id: frame for frame in store.query_frames()
        }
        anchor_timestamps = [
            frames_by_id[frame_id].timestamp
            for frame_id in anchor_frame_ids
            if frame_id in frames_by_id
            and frames_by_id[frame_id].timestamp is not None
        ]
        window_start = max(0.0, start_time - padding)
        window_end = end_time + padding
        timestamps = list(anchor_timestamps)
        current = window_start
        while current <= window_end:
            timestamps.append(current)
            current += interval
        extracted = extract_video_frames_at_timestamps(
            video_path=video_path,
            frame_dir=context['scene_root'] / 'frames' / 'bridge_frames',
            timestamps=timestamps,
        )
        for item in extracted:
            store.add_frame(FrameRecord(
                frame_id=item['frame_id'],
                timestamp=item['timestamp'],
                dataset=context['dataset'],
                scene_name=context['scene_name'],
                video_path=video_path,
                frame_path=item['frame_path'],
                is_keyframe=False,
                metadata={
                    'generated_by': 'find_bridge_frames',
                    'role': EVIDENCE_FRAME_ROLE,
                    'bridge_start': start_time,
                    'bridge_end': end_time,
                    'window_start': window_start,
                    'window_end': window_end,
                    'anchor_frame_ids': anchor_frame_ids,
                },
            ))
        return {
            'bridge_frame_ids': [item['frame_id'] for item in extracted],
            'bridge_frame_paths': [item['frame_path'] for item in extracted],
            'window_start': window_start,
            'window_end': window_end,
            'anchor_frame_ids': anchor_frame_ids,
        }

    def detect_objects(context: Dict[str, Any], **args):
        frame_id = str(args.get('frame_id', '')).strip()
        entity = str(args.get('entity', '')).strip().lower()
        prompt = str(args.get('prompt', '') or entity).strip()
        if not frame_id or not entity:
            return {'error': 'frame_id and entity are required'}
        store = context['store']
        frame = next(
            (item for item in store.query_frames() if item.frame_id == frame_id),
            None,
        )
        if frame is None:
            return {'error': f'Frame not found in memory: {frame_id}'}
        detector = context.get('grounding_dino_detector')
        if detector is None:
            detector = GroundingDinoDetector(device=context['device'])
            context['grounding_dino_detector'] = detector
        image = Image.open(frame.frame_path).convert('RGB')
        candidates = detector.detect_all(image, [prompt]).get(prompt, [])
        for candidate in candidates:
            candidate['label'] = entity
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for index, candidate in enumerate(candidates):
            x1, y1, x2, y2 = [int(value) for value in candidate['bbox']]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=4)
            draw.text((x1 + 3, max(0, y1 + 3)), f'{index}:{entity} {candidate["score"]:.2f}', fill=(255, 255, 0))
        output_dir = context['scene_root'] / 'evidence' / 'detection_candidates'
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f'{frame_id}_{entity}_candidates.jpg'
        json_path = output_dir / f'{frame_id}_{entity}_candidates.json'
        annotated.save(image_path, quality=92)
        payload = {
            'frame_id': frame_id,
            'entity': entity,
            'candidates': candidates,
            'image_path': str(image_path),
            'ambiguous': len(candidates) > 1,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        return payload

    def select_detection(context: Dict[str, Any], **args):
        frame_id = str(args.get('frame_id', '')).strip()
        entity = str(args.get('entity', '')).strip().lower()
        candidate_index = int(args.get('candidate_index', 0))
        if not frame_id or not entity:
            return {'error': 'frame_id and entity are required'}
        candidates_path = (
            context['scene_root'] / 'evidence' / 'detection_candidates' /
            f'{frame_id}_{entity}_candidates.json'
        )
        if not candidates_path.exists():
            return {'error': f'No candidates found for {frame_id}/{entity}'}
        payload = json.loads(candidates_path.read_text(encoding='utf-8'))
        candidates = payload.get('candidates', [])
        if candidate_index < 0 or candidate_index >= len(candidates):
            return {'error': f'candidate_index out of range: {candidate_index}'}
        selected = candidates[candidate_index]
        overrides_path = context['scene_root'] / 'evidence' / 'detection_overrides.json'
        overrides = read_json(overrides_path, {}) or {}
        overrides.setdefault(frame_id, {})[entity] = {
            'bbox': selected['bbox'],
            'score': selected['score'],
            'candidate_index': candidate_index,
            'source': 'planner_disambiguation',
        }
        overrides_path.write_text(
            json.dumps(overrides, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        return {
            'frame_id': frame_id,
            'entity': entity,
            'selected_index': candidate_index,
            'selected': overrides[frame_id][entity],
            'overrides_path': str(overrides_path),
        }

    def verify_candidate(context: Dict[str, Any], **args):
        frame_id = str(args.get('frame_id', '')).strip()
        entity = str(args.get('entity', '')).strip().lower()
        bbox = args.get('bbox')
        if not frame_id or not entity:
            return {'error': 'frame_id and entity are required'}
        store = context['store']
        frame = next(
            (item for item in store.query_frames() if item.frame_id == frame_id),
            None,
        )
        if frame is None:
            return {'error': f'Frame not found in memory: {frame_id}'}
        if bbox is None:
            detection = (
                (frame.metadata or {})
                .get('detections', {})
                .get(entity)
            )
            if detection is None:
                return {'error': f'No bbox available for {frame_id}/{entity}'}
            bbox = detection['bbox']
        image = Image.open(frame.frame_path).convert('RGB')
        x1, y1, x2, y2 = [int(value) for value in bbox]
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=5)
        crop = image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))
        prompt = (
            f'Check whether the red box in the full image contains a valid '
            f'"{entity}" suitable for 3D reconstruction. The cropped region is '
            'also shown. Reject if it is a different object category, a partial '
            'fragment, furniture background, reflection, or too ambiguous. '
            'Return one JSON object in a json code block: '
            '{"valid": true/false, "category": "...", "confidence": 0.0-1.0, '
            '"reason": "..."}'
        )
        reserve_api_call(
            'qwen_check_candidate',
            {'frame_id': frame_id, 'entity': entity},
        )
        response = context.get('vlm_client', context['tool_vlm_client']).chat.completions.create(
            model=context.get('vlm_model', context['model']),
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_to_data_uri(annotated)}},
                    {'type': 'image_url', 'image_url': {'url': image_to_data_uri(crop)}},
                ],
            }],
            max_tokens=512,
            temperature=0.0,
            top_p=0.95,
        )
        raw_response = response.choices[0].message.content
        try:
            result, _ = parse_json_str(raw_response)
        except Exception as exc:
            return {
                'error': f'Failed to parse checker response: {exc}',
                'raw_response': raw_response,
            }
        result = {
            **result,
            'frame_id': frame_id,
            'entity': entity,
            'bbox': bbox,
            'raw_response': raw_response,
        }
        output_dir = context['scene_root'] / 'evidence' / 'candidate_checks'
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f'{frame_id}_{entity}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        return result


    def check_evidence(context: Dict[str, Any]):
        path = context['scene_root'] / 'evidence' / 'evidence_status.json'
        status = read_json(path)
        return {
            'evidence_status': status,
            'path': str(path),
            'exists': status is not None,
        }

    def find_temporal_neighbors(context: Dict[str, Any], **args):
        entity = args.get('entity')
        anchor_ids = args.get('anchor_frame_ids', [])
        offsets = args.get('offsets_seconds', [-1.0, -0.5, 0.5, 1.0])
        max_per_anchor = int(args.get('max_frames_per_anchor', 4))
        store = context['store']
        all_frames = store.query_frames()
        frames_by_id = {frame.frame_id: frame for frame in all_frames}
        anchors = [frames_by_id[item] for item in anchor_ids if item in frames_by_id]
        if not anchors:
            return {'error': 'No valid anchor frame IDs', 'anchor_frame_ids': anchor_ids}

        timestamps = []
        for anchor in anchors:
            values = build_neighbor_timestamps(
                [anchor.timestamp],
                offsets_seconds=tuple(offsets),
                min_timestamp=min(frame.timestamp for frame in all_frames if frame.timestamp is not None),
                max_timestamp=max(frame.timestamp for frame in all_frames if frame.timestamp is not None),
            )
            timestamps.extend(values[:max_per_anchor])
        timestamps = sorted(set(timestamps))

        video_path = next(frame.video_path for frame in all_frames if frame.video_path)
        extracted = extract_video_frames_at_timestamps(
            video_path=video_path,
            frame_dir=context['scene_root'] / 'frames' / 'neighbors',
            timestamps=timestamps,
        )
        for item in extracted:
            store.add_frame(FrameRecord(
                frame_id=item['frame_id'],
                timestamp=item['timestamp'],
                dataset=context['dataset'],
                scene_name=context['scene_name'],
                video_path=video_path,
                frame_path=item['frame_path'],
                is_keyframe=False,
                metadata={
                    'generated_by': 'planner_tool',
                    'role': EVIDENCE_FRAME_ROLE,
                    'entity': entity,
                    'anchor_frame_ids': anchor_ids,
                },
            ))
        return {
            'entity': entity,
            'candidate_frame_ids': [item['frame_id'] for item in extracted],
            'candidate_frame_paths': [item['frame_path'] for item in extracted],
        }

    def collect_question_evidence(context: Dict[str, Any], **args):
        command = [
            sys.executable,
            '-m',
            'entrypoints.collect_vsibench_evidence',
            '--dataset', context['dataset'],
            '--scene-name', context['scene_name'],
            '--question-id', str(context['question_id']),
            '--data-root', str(context['data_root']),
            '--results-root', str(context['results_root']),
            '--detector', str(args.get('detector', 'grounding_dino')),
            '--top-k', str(args.get('top_k', 6)),
            '--min-temporal-gap', str(args.get('min_temporal_gap', 5)),
            '--min-joint-visibility', str(args.get('min_joint_visibility', 0.25)),
            '--min-support-frames', str(args.get('min_support_frames', 2)),
            '--min-visibility', str(args.get('min_visibility', 0.2)),
            '--vlm-workers', str(args.get('vlm_workers', 2)),
        ]
        result = run_subprocess(command, cwd=context['repo_root'])
        status_path = context['scene_root'] / 'evidence' / 'evidence_status.json'
        response = {
            **result,
            'evidence_status': read_json(status_path),
        }
        if result['returncode'] != 0:
            response['error'] = 'collect_question_evidence failed'
        return response

    def estimate_metric_scale_and_distance(context: Dict[str, Any], **args):
        if context.get('question_type') == 'object_size_estimation':
            return {
                'error': (
                    'estimate_metric_scale_and_distance is for distance '
                    'questions. Use estimate_object_size for size questions.'
                ),
            }
        command = [
            sys.executable,
            '-m',
            'entrypoints.estimate_vsibench_metric_scale',
            '--dataset', context['dataset'],
            '--scene-name', context['scene_name'],
            '--question-id', str(context['question_id']),
            '--data-root', str(context['data_root']),
            '--results-root', str(context['results_root']),
            '--device', context.get('device', 'cuda'),
            '--resolution-level', str(max(1, min(int(args.get('resolution_level', 7)), 9))),
        ]
        result = run_subprocess(command, cwd=context['repo_root'])
        evidence_dir = context['scene_root'] / 'evidence'
        response = {
            **result,
            'metric_scale': read_json(evidence_dir / 'metric_scale.json'),
            'distance_result': read_json(evidence_dir / 'distance_result.json'),
        }
        if result['returncode'] != 0:
            response['error'] = 'estimate_metric_scale_and_distance failed'
        return response

    def estimate_object_size(context: Dict[str, Any], **args):
        entity = str(args.get('entity', '')).strip().lower()
        if not entity:
            return {'error': 'entity must not be empty'}
        command = [
            sys.executable,
            '-m',
            'entrypoints.estimate_vsibench_object_size',
            '--dataset', context['dataset'],
            '--scene-name', context['scene_name'],
            '--question-id', str(context['question_id']),
            '--entity', entity,
            '--track-id', str(args.get('track_id', '') or ''),
            '--method', str(args.get('method', 'percentile')),
            '--lower-percentile', str(args.get('lower_percentile', 5.0)),
            '--upper-percentile', str(args.get('upper_percentile', 95.0)),
            '--data-root', str(context['data_root']),
            '--results-root', str(context['results_root']),
            '--device', context.get('device', 'cuda'),
            '--resolution-level', str(max(1, min(int(args.get('resolution_level', 7)), 9))),
        ]
        result = run_subprocess(command, cwd=context['repo_root'])
        evidence_dir = context['scene_root'] / 'evidence'
        response = {
            **result,
            'object_size_result': read_json(evidence_dir / 'object_size_result.json'),
        }
        if result['returncode'] != 0:
            response['error'] = 'estimate_object_size failed'
        return response

    def count_entities_in_video(context: Dict[str, Any], **args):
        entity = str(args.get('entity', '')).strip().lower()
        if not entity:
            return {'error': 'entity must not be empty'}
        max_frames = max(2, int(args.get('max_frames', 12)))
        store = context['store']

        objects = [
            obj for obj in store.query_objects()
            if obj.category.strip().lower() == entity
        ]
        observations = []
        for obj in objects:
            observations.extend(store.query_observations(object_id=obj.object_id))
        observations = [
            observation for observation in observations
            if observation.position_3d is not None
        ]
        explicit_tracks = [
            track for track in store.query_tracks(category=entity)
            if track.get('explicit_track')
        ]
        reliable_tracks = [
            track for track in explicit_tracks
            if (
                len(track.get('frame_ids', [])) >= 2
                or (
                    int(track.get('total_points', 0)) >= 5000
                    and float(track.get('mean_confidence') or 0.0) >= 0.70
                )
            )
        ]
        if reliable_tracks:
            explicit_tracks = reliable_tracks
        if explicit_tracks and args.get('verify_tracks', True):
            verified_tracks = []
            for track in explicit_tracks:
                support = len(track.get('frame_ids', []))
                mean_confidence = float(track.get('mean_confidence') or 0.0)
                suspicious = support < 2 or mean_confidence < 0.70
                if not suspicious:
                    verified_tracks.append(track)
                    continue
                checks = [
                    verify_candidate(
                        context,
                        frame_id=frame_id,
                        entity=entity,
                    )
                    for frame_id in track.get('frame_ids', [])[:2]
                ]
                rejected = any(
                    check.get('valid') is False
                    for check in checks
                    if check.get('error') is None
                )
                if not rejected:
                    verified_tracks.append(track)
            if verified_tracks:
                explicit_tracks = verified_tracks
        if explicit_tracks:
            print(
                f'[Counting] Using object_track_memory for {entity} '
                f'with {len(explicit_tracks)} tracks.',
                flush=True,
            )
            result = {
                'count': len(explicit_tracks),
                'instances': [
                    {
                        'description': 'object track',
                        'track_id': track['track_id'],
                        'centroid': track.get('centroid'),
                        'frame_ids': track.get('frame_ids', []),
                        'timestamps': track.get('timestamps', []),
                    }
                    for track in explicit_tracks
                ],
                'confidence': 0.95,
                'reasoning': (
                    f'Used {len(explicit_tracks)} explicit object tracks from Memory.'
                ),
                'entity': entity,
                'method': 'object_track_memory',
                'frame_ids': [
                    frame_id
                    for track in explicit_tracks
                    for frame_id in track.get('frame_ids', [])
                ],
                'timestamps': [
                    timestamp
                    for track in explicit_tracks
                    for timestamp in track.get('timestamps', [])
                ],
            }
            evidence_dir = context['scene_root'] / 'evidence'
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / 'count_result.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            return result
        if len(observations) >= 2:
            print(
                f'[Counting] Using 3D centroid clustering for {entity} '
                f'with {len(observations)} observations.',
                flush=True,
            )
            positions = np.asarray(
                [observation.position_3d for observation in observations],
                dtype=np.float32,
            )
            extents = []
            for obj in objects:
                extent = (obj.metadata or {}).get('point_extent')
                if extent:
                    extents.append(float(np.linalg.norm(np.asarray(extent))))
            object_scale = max(extents) if extents else 1.0
            eps = max(0.05, 0.2 * object_scale)

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
            for index, observation in enumerate(observations):
                root = find(index)
                clusters.setdefault(root, []).append(observation)

            instances = []
            for cluster_observations in clusters.values():
                centroid = np.mean(
                    [observation.position_3d for observation in cluster_observations],
                    axis=0,
                )
                instances.append({
                    'description': '3D-consistent object instance',
                    'centroid': centroid.astype(float).tolist(),
                    'frame_ids': [
                        observation.frame_id for observation in cluster_observations
                    ],
                    'timestamps': [
                        observation.timestamp for observation in cluster_observations
                    ],
                })
            result = {
                'count': len(instances),
                'instances': instances,
                'confidence': min(1.0, 0.5 + 0.1 * len(observations)),
                'reasoning': (
                    f'Clustered {len(observations)} per-frame 3D object centroids '
                    f'with eps={eps:.3f} in VGGT world coordinates.'
                ),
                'entity': entity,
                'method': '3d_centroid_clustering',
                'eps': eps,
                'object_scale': object_scale,
                'frame_ids': [observation.frame_id for observation in observations],
                'timestamps': [observation.timestamp for observation in observations],
            }
            evidence_dir = context['scene_root'] / 'evidence'
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / 'count_result.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            return result

        memory_frames = [
            frame for frame in store.query_frames()
            if frame.timestamp is not None
            and frame.frame_path
            and Path(frame.frame_path).exists()
        ]
        if not memory_frames:
            video_path = str(
                context['data_root'] / 'videos' / context['dataset'] /
                f"{context['scene_name']}.mp4"
            )
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                return {'error': f'Failed to open video: {video_path}'}
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if fps <= 0 or frame_count <= 0:
                return {'error': f'Invalid video metadata: {video_path}'}
            duration = frame_count / fps
            stride = max(1.0, duration / max_frames)
            timestamps = [
                index * stride
                for index in range(max_frames + 1)
                if index * stride <= duration
            ]
            extracted = extract_video_frames_at_timestamps(
                video_path=video_path,
                frame_dir=context['scene_root'] / 'frames' / 'count_scan',
                timestamps=timestamps,
            )
            memory_frames = [
                FrameRecord(
                    frame_id=item['frame_id'],
                    timestamp=item['timestamp'],
                    dataset=context['dataset'],
                    scene_name=context['scene_name'],
                    video_path=video_path,
                    frame_path=item['frame_path'],
                    is_keyframe=False,
                    metadata={'generated_by': 'count_entities_in_video'},
                )
                for item in extracted
            ]

        visible_frames = []
        for frame in memory_frames:
            metadata = frame.metadata or {}
            visibility = metadata.get('visibility', {})
            detections = metadata.get('detections', {})
            if float(visibility.get(entity, 0.0)) > 0.0 or entity in detections:
                visible_frames.append(frame)
        if visible_frames:
            memory_frames = visible_frames

        if os.environ.get('GCA_ALLOW_QWEN_COUNTING', '0') != '1':
            return {
                'error': (
                    'Counting requires 3D object tracks/observations. '
                    'Qwen VLM counting fallback is disabled; set '
                    'GCA_ALLOW_QWEN_COUNTING=1 to enable it explicitly.'
                ),
                'entity': entity,
                'method': 'missing_3d_tracks',
            }

        print(
            f'[Counting] Falling back to VLM multi-frame counting for {entity} '
            f'with {len(memory_frames)} candidate frames.',
            flush=True,
        )

        memory_frames.sort(key=lambda frame: frame.timestamp)
        count = min(max_frames, len(memory_frames))
        indices = [
            int(round(value))
            for value in (
                index * (len(memory_frames) - 1) / max(1, count - 1)
                for index in range(count)
            )
        ]
        selected_frames = [
            memory_frames[index] for index in sorted(set(indices))
        ]

        prompt = (
            f'Count the number of unique physical instances of the category '
            f'"{entity}" in this room/video. The images are chronological samples. '
            'Count unique objects, not repeated appearances of the same object in '
            'multiple frames. Do not count images, rugs, tables, or other categories. '
            'Return exactly one JSON object with this schema: '
            '{"count": integer, "instances": [{"description": "...", '
            '"evidence_frame_indices": [0, 1]}], "confidence": 0.0-1.0, '
            '"reasoning": "..."}'
        )
        content = [{'type': 'text', 'text': prompt}]
        for frame_index, frame in enumerate(selected_frames):
            content.append({
                'type': 'text',
                'text': (
                    f'Frame {frame_index}: frame_id={frame.frame_id}, '
                    f'timestamp={frame.timestamp:.2f}s'
                ),
            })
            image = Image.open(frame.frame_path).convert('RGB')
            content.append({
                'type': 'image_url',
                'image_url': {'url': image_to_data_uri(image)},
            })

        reserve_api_call('count_entities_in_video', {'entity': entity})
        response = context.get('vlm_client', context['tool_vlm_client']).chat.completions.create(
            model=context.get('vlm_model', context['model']),
            messages=[{'role': 'user', 'content': content}],
            max_tokens=2048,
            temperature=0.0,
            top_p=0.95,
        )
        raw_content = response.choices[0].message.content
        try:
            result, _ = parse_json_str(raw_content)
        except Exception as exc:
            return {
                'error': f'Failed to parse counting response: {exc}',
                'raw_response': raw_content,
            }

        result = {
            **result,
            'entity': entity,
            'method': 'vlm_multi_frame',
            'frame_ids': [frame.frame_id for frame in selected_frames],
            'timestamps': [frame.timestamp for frame in selected_frames],
            'raw_response': raw_content,
        }
        evidence_dir = context['scene_root'] / 'evidence'
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / 'count_result.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        return result

    def query_tracks(context: Dict[str, Any], **args):
        category = args.get('category')
        if category is not None:
            category = str(category).strip().lower()
        tracks = context['store'].query_tracks(category=category)
        return {
            'tracks': [
                {
                    'track_id': track['track_id'],
                    'category': track.get('category'),
                    'support': len(track.get('frame_ids', [])),
                    'total_points': track.get('total_points', 0),
                    'mean_confidence': track.get('mean_confidence'),
                    'centroid': track.get('centroid'),
                    'frame_ids': track.get('frame_ids', []),
                }
                for track in tracks
            ]
        }

    registry.register(ToolSpec(
        name='scan_entity_visibility',
        description=(
            'Scan the video timeline for target visibility using GroundingDINO, '
            'and return representative anchor frames and a suggested bridge window.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'entities': {'type': 'array', 'items': {'type': 'string'}},
                'scan_stride_seconds': {'type': 'number', 'default': 4.0},
                'max_scan_frames': {'type': 'integer', 'default': 80},
            },
            'required': ['entities'],
        },
        handler=scan_entity_visibility,
    ))
    registry.register(ToolSpec(
        name='find_bridge_frames',
        description=(
            'Extract bridge frames across a time interval and include the '
            'representative anchor frames returned by scan_entity_visibility.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'start_time': {'type': 'number'},
                'end_time': {'type': 'number'},
                'interval_seconds': {'type': 'number', 'default': 3.0},
                'padding_seconds': {'type': 'number', 'default': 1.0},
                'anchor_frame_ids': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
            },
            'required': ['start_time', 'end_time'],
        },
        handler=find_bridge_frames,
    ))
    registry.register(ToolSpec(
        name='detect_objects',
        description=(
            'Run GroundingDINO on one frame and return all candidate boxes for '
            'an entity, with an annotated candidate visualization.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'frame_id': {'type': 'string'},
                'entity': {'type': 'string'},
                'prompt': {'type': 'string'},
            },
            'required': ['frame_id', 'entity'],
        },
        handler=detect_objects,
    ))
    registry.register(ToolSpec(
        name='select_detection',
        description=(
            'Select one candidate box returned by detect_objects. The selected '
            'box is stored as a detection override for later reconstruction.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'frame_id': {'type': 'string'},
                'entity': {'type': 'string'},
                'candidate_index': {'type': 'integer', 'default': 0},
            },
            'required': ['frame_id', 'entity', 'candidate_index'],
        },
        handler=select_detection,
    ))
    registry.register(ToolSpec(
        name='verify_candidate',
        description=(
            'Use Qwen only as a checker on one selected candidate. It checks '
            'whether the boxed crop is the requested object. Use this only for '
            'suspicious tracks or representative frames, not every frame.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'frame_id': {'type': 'string'},
                'entity': {'type': 'string'},
                'bbox': {
                    'type': 'array',
                    'items': {'type': 'number'},
                },
            },
            'required': ['frame_id', 'entity'],
        },
        handler=verify_candidate,
    ))
    registry.register(ToolSpec(
        name='check_evidence',
        description='Read the current EvidenceSufficiencyChecker status.',
        parameters={'type': 'object', 'properties': {}},
        handler=check_evidence,
    ))
    registry.register(ToolSpec(
        name='find_temporal_neighbors',
        description='Extract nearby frames around evidence anchor frames for a missing entity.',
        parameters={
            'type': 'object',
            'properties': {
                'entity': {'type': 'string'},
                'anchor_frame_ids': {'type': 'array', 'items': {'type': 'string'}},
                'offsets_seconds': {'type': 'array', 'items': {'type': 'number'}},
                'max_frames_per_anchor': {'type': 'integer', 'default': 4},
            },
            'required': ['entity', 'anchor_frame_ids'],
        },
        handler=find_temporal_neighbors,
    ))
    registry.register(ToolSpec(
        name='collect_question_evidence',
        description=(
            'Run target-conditioned GroundingDINO detection, SAM2 mask generation, '
            'VGGT reconstruction and evidence selection. Qwen is used only for '
            'ambiguous semantic checks. The selector reserves support frames per entity.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'top_k': {'type': 'integer', 'default': 6},
                'min_support_frames': {'type': 'integer', 'default': 2},
                'min_visibility': {'type': 'number', 'default': 0.2},
                'min_temporal_gap': {'type': 'number', 'default': 5.0},
                'min_joint_visibility': {'type': 'number', 'default': 0.25},
                'vlm_workers': {'type': 'integer', 'default': 2},
                'detector': {
                    'type': 'string',
                    'enum': ['grounding_dino', 'vlm'],
                    'default': 'grounding_dino',
                },
            },
        },
        handler=collect_question_evidence,
    ))
    registry.register(ToolSpec(
        name='estimate_metric_scale_and_distance',
        description='Estimate metric scale from VGGT/MoGe and compute closest-point distance for target objects.',
        parameters={
            'type': 'object',
            'properties': {
                'resolution_level': {'type': 'integer', 'default': 7},
            },
        },
        handler=estimate_metric_scale_and_distance,
    ))
    registry.register(ToolSpec(
        name='estimate_object_size',
        description=(
            'Estimate metric scale and the longest dimension of one object '
            'from its 3D point cloud.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'entity': {'type': 'string'},
                'track_id': {'type': 'string'},
                'method': {
                    'type': 'string',
                    'enum': ['raw', 'percentile', 'obb'],
                    'default': 'percentile',
                },
                'lower_percentile': {'type': 'number', 'default': 5.0},
                'upper_percentile': {'type': 'number', 'default': 95.0},
                'resolution_level': {'type': 'integer', 'default': 7},
            },
            'required': ['entity'],
        },
        handler=estimate_object_size,
    ))
    registry.register(ToolSpec(
        name='count_entities_in_video',
        description=(
            'Count unique physical instances of one category from 3D object '
            'tracks. Suspicious tracks may be checked once per representative '
            'frame by Qwen; routine VLM counting is disabled.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'entity': {'type': 'string'},
                'max_frames': {'type': 'integer', 'default': 12},
                'verify_tracks': {'type': 'boolean', 'default': True},
            },
            'required': ['entity'],
        },
        handler=count_entities_in_video,
    ))
    registry.register(ToolSpec(
        name='query_tracks',
        description=(
            'List object tracks in Scene Memory with support, points, '
            'confidence and centroid. Use this to choose a track before size '
            'or geometry estimation.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'category': {'type': 'string'},
            },
        },
        handler=query_tracks,
    ))

    # ------------------------------------------------------------------
    # Executable geometry constraint tools
    # ------------------------------------------------------------------
    def _constraint_and_bindings(context: Dict[str, Any], args: Dict[str, Any]):
        constraint = load_constraint(context['scene_root'], context['question_id'])
        if constraint is None:
            return None, None, {'error': 'No task constraint compiled for this question'}
        stored = load_bindings(context['scene_root'], context['question_id'])
        bindings = merge_bindings(constraint, stored, args.get('bindings'))
        return constraint, bindings, None

    def get_task_constraint(context: Dict[str, Any], **args):
        constraint, bindings, error = _constraint_and_bindings(context, args)
        if error:
            return error
        store_root = context['store'].root_dir
        points = resolve_points(
            store_root,
            (constraint.get('operation') or {}).get('operation'),
            bindings,
            object_ids=[obj.object_id for obj in context['store'].query_objects()],
        )
        metric_scale = read_json(context['scene_root'] / 'evidence' / 'metric_scale.json', {}) or {}
        frame_id = args.get('coordinate_frame_id') or f"vggt_world_q{context['question_id']}"
        ctx = build_constraint_context(
            constraint,
            context['scene_root'],
            context['question_id'],
            store_root=store_root,
            bindings=bindings,
            points=points,
            metric_scale=metric_scale,
            coordinate_frame_id=frame_id,
        )
        report = validate_op(constraint, ctx)
        return {
            'constraint_path': str(
                context['scene_root'] / 'questions' / str(context['question_id']) / 'task_constraint.json'
            ),
            'operation': (constraint.get('operation') or {}).get('operation'),
            'unit': (constraint.get('operation') or {}).get('unit'),
            'options': (constraint.get('operation') or {}).get('options'),
            'entity_roles': [
                {'role': item.get('role'), 'category': item.get('category')}
                for item in constraint.get('entities', [])
            ],
            'reference_frame': constraint.get('reference_frame'),
            'bindings': bindings,
            'resolved_roles': sorted(points.keys()),
            'metric_scale': metric_scale,
            'validation': report,
        }

    def bind_constraint_entities(context: Dict[str, Any], **args):
        constraint = load_constraint(context['scene_root'], context['question_id'])
        if constraint is None:
            return {'error': 'No task constraint compiled for this question'}
        incoming = args.get('bindings') or {}
        if not isinstance(incoming, dict) or not incoming:
            return {'error': 'bindings must be a non-empty object'}
        stored = load_bindings(context['scene_root'], context['question_id'])
        merged = merge_bindings(constraint, stored, incoming)
        path = save_bindings(context['scene_root'], context['question_id'], merged)
        known_ids = {obj.object_id for obj in context['store'].query_objects()}
        unknown = {
            role: value
            for role, value in merged.items()
            if isinstance(value, str) and value and value not in known_ids
        }
        return {
            'bindings': merged,
            'bindings_path': str(path),
            'known_object_ids': sorted(known_ids),
            'unknown_instances': unknown,
        }

    def validate_operation_tool(context: Dict[str, Any], **args):
        constraint, bindings, error = _constraint_and_bindings(context, args)
        if error:
            return error
        store_root = context['store'].root_dir
        operation = args.get('operation') or (constraint.get('operation') or {}).get('operation')
        points = resolve_points(
            store_root,
            operation,
            bindings,
            object_ids=[obj.object_id for obj in context['store'].query_objects()],
        )
        metric_scale = read_json(context['scene_root'] / 'evidence' / 'metric_scale.json', {}) or {}
        frame_id = args.get('coordinate_frame_id') or f"vggt_world_q{context['question_id']}"
        ctx = build_constraint_context(
            constraint,
            context['scene_root'],
            context['question_id'],
            store_root=store_root,
            bindings=bindings,
            points=points,
            metric_scale=metric_scale,
            coordinate_frame_id=frame_id,
        )
        report = validate_op(constraint, ctx)
        return {
            'operation': operation,
            'bindings': bindings,
            'resolved_roles': sorted(points.keys()),
            'validation': report,
        }

    def execute_operation_tool(context: Dict[str, Any], **args):
        constraint, bindings, error = _constraint_and_bindings(context, args)
        if error:
            return error
        store = context['store']
        store_root = store.root_dir
        operation = args.get('operation') or (constraint.get('operation') or {}).get('operation')
        object_ids = [obj.object_id for obj in store.query_objects()]
        points = resolve_points(store_root, operation, bindings, object_ids=object_ids)
        metric_scale = read_json(context['scene_root'] / 'evidence' / 'metric_scale.json', {}) or {}
        frame_id = args.get('coordinate_frame_id') or f"vggt_world_q{context['question_id']}"
        ctx = build_constraint_context(
            constraint,
            context['scene_root'],
            context['question_id'],
            store_root=store_root,
            bindings=bindings,
            points=points,
            metric_scale=metric_scale,
            coordinate_frame_id=frame_id,
        )
        extra = resolve_extra_inputs(operation, bindings, args, store=store)
        inputs = {
            'points': points,
            'bindings': bindings,
            'options': (constraint.get('operation') or {}).get('options') or [],
        }
        inputs.update(extra)
        report = run_constraint_operation(
            constraint,
            inputs,
            context=ctx,
            scene_root=context['scene_root'],
            question_id=context['question_id'],
            persist=True,
        )
        if args.get('bindings'):
            save_bindings(
                context['scene_root'],
                context['question_id'],
                merge_bindings(
                    constraint,
                    load_bindings(context['scene_root'], context['question_id']),
                    args.get('bindings'),
                ),
            )
        return report

    def validate_result_tool(context: Dict[str, Any], **args):
        store = load_result_store(context['scene_root'], context['question_id'])
        operation_id = str(args.get('operation_result_id', '')).strip()
        if not operation_id:
            return {
                'error': 'operation_result_id is required',
                'available_operation_results': sorted(store.all().keys()),
            }
        entry = store.get(operation_id)
        if entry is None:
            return {
                'error': f'Unknown operation_result_id: {operation_id}',
                'available_operation_results': sorted(store.all().keys()),
            }
        constraint = load_constraint(context['scene_root'], context['question_id'])
        report = validate_res(constraint, entry, {})
        verification = {
            'operation_result_id': operation_id,
            'verification_status': 'verified' if report['valid'] else 'rejected',
            'validation': report,
            'value': entry.get('value'),
            'unit': entry.get('unit'),
            'mapped_answer': report.get('mapped_answer'),
            'quality_flags': entry.get('quality_flags'),
        }
        store.update(operation_id, {
            'verification_status': verification['verification_status'],
            'verification': report,
        })
        return verification

    registry.register(ToolSpec(
        name='get_task_constraint',
        description=(
            'Show the compiled executable task constraint for this question: '
            'operation, required roles, current bindings, resolved evidence and '
            'the structured pre-execution validation report.'
        ),
        parameters={'type': 'object', 'properties': {}},
        handler=get_task_constraint,
    ))
    registry.register(ToolSpec(
        name='bind_constraint_entities',
        description=(
            'Bind semantic constraint roles (origin/forward/target/entity_a/'
            'entity_b/reference_entity/candidate_entities/category) to concrete '
            'instance ids so the geometric operation can execute.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'bindings': {'type': 'object'},
            },
            'required': ['bindings'],
        },
        handler=bind_constraint_entities,
    ))
    registry.register(ToolSpec(
        name='validate_operation',
        description=(
            'Check whether the current constraint may legally execute: entity '
            'binding, coordinate frame, geometry version, metric scale, units and '
            'degeneracy. Returns structured errors with suggested repair actions.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'operation': {'type': 'string'},
                'bindings': {'type': 'object'},
                'coordinate_frame_id': {'type': 'string'},
            },
        },
        handler=validate_operation_tool,
    ))
    registry.register(ToolSpec(
        name='execute_operation',
        description=(
            'Run the fixed geometric operation for this question (surface_distance, '
            'relative_direction, argmin_distance, object_extent, count_instances, '
            'first_visible_order). Validates inputs, computes the value, validates '
            'the result and stores it with an operation_id. Final answers must cite '
            'a verified operation_id.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'operation': {'type': 'string'},
                'bindings': {'type': 'object'},
                'method': {
                    'type': 'string',
                    'enum': ['raw', 'percentile', 'obb'],
                    'default': 'percentile',
                },
                'vertical_axis': {
                    'type': 'array',
                    'items': {'type': 'number'},
                },
                'coordinate_frame_id': {'type': 'string'},
            },
        },
        handler=execute_operation_tool,
    ))
    registry.register(ToolSpec(
        name='validate_result',
        description=(
            'Re-validate one stored operation result by its operation_result_id '
            'and return the final verification status plus mapped answer.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'operation_result_id': {'type': 'string'},
            },
            'required': ['operation_result_id'],
        },
        handler=validate_result_tool,
    ))
    return registry


def build_state_provider(context):
    def state_provider():
        store = context['store']
        evidence_dir = context['scene_root'] / 'evidence'
        memory_frames = store.query_frames()
        candidate_frames = [
            frame for frame in memory_frames
            if is_evidence_candidate_frame(frame)
        ]
        scene_summary = store.summary()
        scene_summary['counts']['evidence_candidate_frames'] = len(candidate_frames)
        scene_summary['counts']['visibility_scan_frames'] = sum(
            (frame.metadata or {}).get('role') == VISIBILITY_SCAN_FRAME_ROLE
            for frame in memory_frames
        )
        scene_summary['counts']['context_frames'] = sum(
            (frame.metadata or {}).get('role', CONTEXT_FRAME_ROLE)
            == CONTEXT_FRAME_ROLE
            for frame in memory_frames
        )
        constraint = context.get('task_constraint') or {}
        question_root = context['scene_root'] / 'questions' / str(context['question_id'])
        stored_bindings = read_json(question_root / 'entity_bindings.json', {}) or {}
        op_results = read_json(question_root / 'operation_results.json', {}) or {}
        verified_results = {
            key: {
                'operation': value.get('operation'),
                'value': value.get('value'),
                'unit': value.get('unit'),
                'quality_flags': value.get('quality_flags'),
            }
            for key, value in op_results.items()
            if value.get('verification_status') == 'verified'
        }
        constraint_validation = {}
        if constraint:
            try:
                bindings = merge_bindings(constraint, stored_bindings)
                operation = (constraint.get('operation') or {}).get('operation')
                points = resolve_points(
                    store.root_dir,
                    operation,
                    bindings,
                    object_ids=[obj.object_id for obj in store.query_objects()],
                )
                metric_scale = read_json(evidence_dir / 'metric_scale.json', {}) or {}
                constraint_validation = validate_op(
                    constraint,
                    build_constraint_context(
                        constraint,
                        context['scene_root'],
                        context['question_id'],
                        store_root=store.root_dir,
                        bindings=bindings,
                        points=points,
                        metric_scale=metric_scale,
                        coordinate_frame_id=f"vggt_world_q{context['question_id']}",
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                constraint_validation = {'valid': False, 'error': str(exc)}
        return {
            'scene_summary': scene_summary,
            'task_constraint': {
                'operation': (constraint.get('operation') or {}).get('operation'),
                'unit': (constraint.get('operation') or {}).get('unit'),
                'options': (constraint.get('operation') or {}).get('options'),
                'entity_roles': [
                    {'role': item.get('role'), 'category': item.get('category')}
                    for item in constraint.get('entities', [])
                ],
                'reference_frame': constraint.get('reference_frame'),
            } if constraint else {},
            'constraint_bindings': stored_bindings,
            'constraint_validation': constraint_validation,
            'operation_results': {
                key: {
                    'operation': value.get('operation'),
                    'value': value.get('value'),
                    'unit': value.get('unit'),
                    'quality_flags': value.get('quality_flags'),
                    'verification_status': value.get('verification_status'),
                }
                for key, value in op_results.items()
            },
            'verified_operation_results': verified_results,
            'evidence_status': read_json(evidence_dir / 'evidence_status.json', {}),
            'next_evidence_request': read_json(evidence_dir / 'next_evidence_request.json', {}),
            'metric_scale': read_json(evidence_dir / 'metric_scale.json', {}),
            'distance_result': read_json(evidence_dir / 'distance_result.json', {}),
            'object_size_result': read_json(evidence_dir / 'object_size_result.json', {}),
            'count_result': read_json(evidence_dir / 'count_result.json', {}),
        }
    return state_provider


def load_question(data_root: Path, dataset: str, scene_name: str, question_id: int):
    jsonl_path = data_root / 'test.jsonl'
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row['dataset'] == dataset
                and row['scene_name'] == scene_name
                and row['id'] == question_id
            ):
                return row
    raise ValueError(
        f'Question {question_id} not found for {dataset}/{scene_name}'
    )


def sanitize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    meta_entities = {'room', 'scene', 'place', 'environment', 'area'}
    sanitized = dict(plan)
    sanitized['reference_entities'] = [
        entity for entity in sanitized.get('reference_entities', [])
        if str(entity).strip().lower() not in meta_entities
    ]
    reference_frame = str(sanitized.get('reference_frame', '')).strip().lower()
    if reference_frame.startswith('object:'):
        entity = reference_frame.split(':', 1)[1].strip().lower()
        if entity in meta_entities:
            sanitized['reference_frame'] = 'world'
    return sanitized


def make_done_validator(context):
    """Gate the Planner's final answer on a verified operation result."""

    def validator(decision: Dict[str, Any]) -> Dict[str, Any]:
        constraint = context.get('task_constraint') or {}
        declared_operation = (constraint.get('operation') or {}).get('operation')
        if not declared_operation:
            return {
                'accepted': True,
                'gate': 'disabled',
                'reason': (
                    'No executable operation is defined for this question type; '
                    'accepting the free-form answer.'
                ),
            }
        store = load_result_store(context['scene_root'], context['question_id'])
        results = store.all()
        op_id = (
            decision.get('operation_result_id')
            or decision.get('operation_id')
            or ''
        )
        op_id = str(op_id).strip()
        verified = [
            key for key, value in results.items()
            if value.get('verification_status') == 'verified'
        ]
        if not op_id:
            if len(verified) == 1:
                op_id = verified[0]
            else:
                return {
                    'accepted': False,
                    'reason': (
                        'A final answer must cite a verified operation_result_id. '
                        'Call execute_operation, then finalize with its operation_id.'
                    ),
                    'available_operation_results': sorted(results),
                    'verified_operation_results': verified,
                    'suggested_actions': ['execute_operation'],
                }
        entry = results.get(op_id)
        if entry is None:
            return {
                'accepted': False,
                'reason': f'Unknown operation_result_id: {op_id}',
                'available_operation_results': sorted(results),
                'verified_operation_results': verified,
            }
        if entry.get('verification_status') != 'verified':
            return {
                'accepted': False,
                'reason': (
                    f'Operation result {op_id} is not verified; repair the '
                    'reported errors before finalizing.'
                ),
                'operation_result_id': op_id,
                'errors': (entry.get('verification') or {}).get('errors', []),
                'quality_flags': entry.get('quality_flags'),
                'suggested_actions': [
                    error.get('suggested_actions', ['execute_operation'])
                    for error in (entry.get('verification') or {}).get('errors', [])
                ],
            }
        report = validate_res(context.get('task_constraint') or {}, entry, {})
        if not report['valid']:
            return {
                'accepted': False,
                'reason': f'Operation result {op_id} failed re-validation.',
                'operation_result_id': op_id,
                'errors': report['errors'],
            }
        verified_answer = report.get('mapped_answer')
        if verified_answer is None:
            verified_answer = entry.get('value')
        return {
            'accepted': True,
            'operation_result_id': op_id,
            'verified_value': entry.get('value'),
            'unit': entry.get('unit'),
            'final_answer': str(verified_answer),
        }

    return validator


async def main():
    args = parse_args()
    if args.max_turns is not None:
        args.max_steps_per_round = args.max_turns
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    repo_root = Path(__file__).resolve().parents[1]
    scene_root = results_root / args.dataset / args.scene_name
    scene_root.mkdir(parents=True, exist_ok=True)
    configure_api_budget(
        scene_root / 'memory' / 'api_budget.json',
        max_calls=args.max_api_calls,
        reset=True,
    )
    print(
        f'[ApiBudget] Max Qwen API calls for this run: {args.max_api_calls}',
        flush=True,
    )
    question = load_question(
        data_root, args.dataset, args.scene_name, args.question_id
    )

    planner_endpoint = resolve_endpoint('planner')
    vlm_endpoint = resolve_endpoint('vlm')
    planner_client = create_async_client(planner_endpoint)
    vlm_client = create_sync_client(vlm_endpoint)
    model = planner_endpoint.model
    base_url = vlm_endpoint.base_url
    api_key = vlm_endpoint.api_key
    sync_client = vlm_client
    client = planner_client
    planner_model = planner_endpoint.model
    vlm_model = vlm_endpoint.model
    print(f'[Agent] Planner endpoint: {json.dumps(planner_endpoint.describe())}', flush=True)
    print(f'[Agent] VLM endpoint    : {json.dumps(vlm_endpoint.describe())}', flush=True)
    if planner_endpoint.model == vlm_endpoint.model and (
        planner_endpoint.base_url == vlm_endpoint.base_url
    ):
        print(
            '[Agent] Planner and VLM share one model; set AGENT_PLANNER_* to '
            'use a cheaper text-only model for planning.',
            flush=True,
        )

    print(f'[Agent] Question: {question["question"]}', flush=True)
    print(f'[Agent] Scene root: {scene_root}', flush=True)
    plan_path = scene_root / 'question_plans' / f'{args.question_id}.json'
    if plan_path.exists() and not args.force_plan:
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        print(f'[Agent] Loaded existing Evidence Plan: {plan_path}', flush=True)
    else:
        evidence_planner = QuestionEvidencePlanner(client=client, model=model)
        try:
            evidence_request = await evidence_planner.plan(
                question=question['question'],
                question_type=question['question_type'],
                options=question['options'],
            )
        except ApiBudgetExceeded as exc:
            evidence_dir = scene_root / 'evidence'
            evidence_dir.mkdir(parents=True, exist_ok=True)
            result = {
                'done': False,
                'error': str(exc),
                'api_budget': get_api_budget(),
            }
            (evidence_dir / 'agent_result.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return
        plan = evidence_request.to_dict()
        plan.update({
            'question_id': question['id'],
            'dataset': question['dataset'],
            'scene_name': question['scene_name'],
            'question_type': question['question_type'],
            'question': question['question'],
            'options': question['options'],
            'ground_truth': question['ground_truth'],
        })
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(f'[Agent] Generated Evidence Plan: {plan_path}', flush=True)
    planner_plan = sanitize_plan({
        key: value for key, value in plan.items() if key != 'ground_truth'
    })
    task_constraint = compile_task_constraint(planner_plan)
    constraint_file = save_constraint(task_constraint, scene_root, args.question_id)
    task_constraint_dict = task_constraint.to_dict()
    print(
        f'[Constraint] Compiled executable task constraint: {constraint_file}',
        flush=True,
    )
    print(
        '[Constraint] operation='
        f"{task_constraint_dict['operation']['operation']!r} "
        f"unit={task_constraint_dict['operation']['unit']!r} roles="
        f"{[item['role'] + ':' + item['category'] for item in task_constraint_dict['entities']]}",
        flush=True,
    )
    if task_constraint_dict['operation']['operation'] is None:
        print(
            '[Constraint] WARNING: no executable operation is defined for this '
            'question type; the Planner falls back to open-ended tool use.',
            flush=True,
        )

    store = open_vsibench_memory(
        data_root=data_root,
        dataset=args.dataset,
        scene_name=args.scene_name,
        scene_root=scene_root,
    )
    scene_metadata_path = store.root_dir / 'scene.json'
    if not scene_metadata_path.exists():
        scene_metadata_path.write_text(
            json.dumps({
                'dataset': args.dataset,
                'scene_name': args.scene_name,
                'video_path': str(
                    data_root / 'videos' / args.dataset / f'{args.scene_name}.mp4'
                ),
            }, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    evidence_dir = scene_root / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    agent_memory_path = evidence_dir / 'agent_memory.json'
    if args.reset_agent_memory and agent_memory_path.exists():
        import time
        backup_path = evidence_dir / f'agent_memory.failed.{int(time.time())}.json'
        agent_memory_path.rename(backup_path)
        print(f'[Agent] Reset Agent Memory. Backup: {backup_path}', flush=True)
    agent_memory = AgentMemory.load(agent_memory_path) if agent_memory_path.exists() else AgentMemory()

    registry = register_tools(ToolRegistry())
    if args.allow_tool_generation:
        for generated_spec in load_generated_tool_specs(
            repo_root / 'tools' / 'generated'
        ):
            if generated_spec.name not in {
                spec.name for spec in registry.list_specs()
            }:
                registry.register(generated_spec)
                print(
                    f'[ToolRegistry] Loaded generated tool: {generated_spec.name}',
                    flush=True,
                )
        print('[ToolRegistry] Tool generation ENABLED.', flush=True)
    else:
        print(
            '[ToolRegistry] Tool generation disabled: fixed tool set for the '
            'core constraint experiment (pass --allow-tool-generation to enable).',
            flush=True,
        )
    context = {
        'repo_root': repo_root,
        'data_root': data_root,
        'results_root': results_root,
        'dataset': args.dataset,
        'scene_name': args.scene_name,
        'question_id': args.question_id,
        'question_type': question['question_type'],
        'scene_root': scene_root,
        'task_constraint': task_constraint_dict,
        'store': store,
        'device': args.device,
        'tool_vlm_client': vlm_client,
        'vlm_client': vlm_client,
        'vlm_model': vlm_model,
        'planner_model': planner_model,
        'model': vlm_model,
        'base_url': vlm_endpoint.base_url,
        'api_key': vlm_endpoint.api_key,
        'llm_endpoints': {
            'planner': planner_endpoint.describe(),
            'vlm': vlm_endpoint.describe(),
        },
    }
    tool_synthesizer = (
        ToolSynthesizer(
            client=client,
            model=model,
            generated_dir=repo_root / 'tools' / 'generated',
        )
        if args.allow_tool_generation
        else None
    )
    loop = PlannerLoop(
        client=client,
        model=model,
        registry=registry,
        context=context,
        agent_memory=agent_memory,
        max_rounds=args.max_rounds,
        max_steps_per_round=args.max_steps_per_round,
        max_total_steps=args.max_total_steps,
        max_tool_failures=args.max_tool_failures,
        state_provider=build_state_provider(context),
        tool_synthesizer=tool_synthesizer,
        done_validator=make_done_validator(context),
    )
    print('[Agent] Starting Planner loop...', flush=True)
    result = await loop.run(question=plan['question'], plan=planner_plan)
    agent_memory.save(agent_memory_path)
    result['llm_endpoints'] = {
        'planner': planner_endpoint.describe(),
        'vlm': vlm_endpoint.describe(),
    }
    result['task_constraint'] = task_constraint_dict
    result['constraint_dir'] = str(
        scene_root / 'questions' / str(args.question_id)
    )
    result['operation_results'] = {
        key: {
            'operation': value.get('operation'),
            'value': value.get('value'),
            'unit': value.get('unit'),
            'quality_flags': value.get('quality_flags'),
            'verification_status': value.get('verification_status'),
        }
        for key, value in load_result_store(
            scene_root, args.question_id
        ).all().items()
    }
    result['api_budget'] = get_api_budget()
    result_path = evidence_dir / 'agent_result.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Agent result saved to: {result_path}')
    print(
        f"[ApiBudget] Used {result['api_budget'].get('used', 0)}/"
        f"{result['api_budget'].get('max_calls', 0)} Qwen API calls.",
        flush=True,
    )


if __name__ == '__main__':
    asyncio.run(main())
