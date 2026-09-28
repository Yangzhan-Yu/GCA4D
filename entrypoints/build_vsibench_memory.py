import argparse
import json
from pathlib import Path

from evals.vsibench import extract_uniform_frames
from tools.apis.four_d_memory import (
    CONTEXT_FRAME_ROLE,
    Event,
    FrameRecord,
    open_vsibench_memory,
)


def parse_args():
    parser = argparse.ArgumentParser('Build VSI-Bench scene memory')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument(
        '--data-root',
        default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench',
    )
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_scene_rows(data_root: Path, dataset: str, scene_name: str):
    jsonl_path = data_root / 'test.jsonl'
    rows = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['dataset'] == dataset and row['scene_name'] == scene_name:
                rows.append(row)
    if not rows:
        raise ValueError(f'No questions found for {dataset}/{scene_name}')
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    video_path = data_root / 'videos' / args.dataset / f'{args.scene_name}.mp4'
    results_root = Path(args.results_root)
    scene_root = results_root / args.dataset / args.scene_name
    frame_dir = scene_root / 'frames'
    rows = load_scene_rows(data_root, args.dataset, args.scene_name)

    print(f'scene: {args.dataset}/{args.scene_name}')
    print(f'questions: {len(rows)}')
    print(f'video: {video_path}')

    if args.dry_run:
        print('dry_run: OK')
        return

    frames = extract_uniform_frames(
        str(video_path), str(frame_dir), args.num_frames
    )

    store = open_vsibench_memory(
        data_root=data_root,
        dataset=args.dataset,
        scene_name=args.scene_name,
        scene_root=scene_root,
    )
    for frame_index, frame_path, timestamp in frames:
        store.add_frame(FrameRecord(
            frame_id=f'{frame_index:06d}',
            timestamp=timestamp,
            dataset=args.dataset,
            scene_name=args.scene_name,
            video_path=str(video_path),
            frame_path=frame_path,
            is_keyframe=True,
            metadata={
                'frame_index': frame_index,
                'role': CONTEXT_FRAME_ROLE,
                'generated_by': 'build_scene_memory_uniform',
            },
        ))

    store.add_event(Event(
        event_id='scene_loaded',
        event_type='scene',
        label='scene_loaded',
        time_start=frames[0][2],
        time_end=frames[-1][2],
        frame_id=frames[0][0],
        confidence=1.0,
        metadata={'question_count': len(rows)},
    ))

    (store.root_dir / 'scene.json').write_text(
        json.dumps({
            'dataset': args.dataset,
            'scene_name': args.scene_name,
            'video_path': str(video_path),
            'num_questions': len(rows),
            'question_ids': [row['id'] for row in rows],
            'question_types': sorted({row['question_type'] for row in rows}),
        }, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    exported = store.export_parquet()
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print('parquet:', json.dumps(exported, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
