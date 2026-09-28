from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from evals.base import BaseBenchmark, BaseBenchmarkSample


MCA_QUESTION_TYPES = [
    'object_rel_direction_easy',
    'object_rel_direction_medium',
    'object_rel_direction_hard',
    'object_rel_distance',
    'route_planning',
    'obj_appearance_order',
]

NA_QUESTION_TYPES = [
    'object_abs_distance',
    'object_counting',
    'object_size_estimation',
    'room_size_estimation',
]

ALL_QUESTION_TYPES = MCA_QUESTION_TYPES + NA_QUESTION_TYPES

CANONICAL_QUESTION_TYPES = [
    'object_counting',
    'object_abs_distance',
    'object_size_estimation',
    'room_size_estimation',
    'object_rel_distance',
    'object_rel_direction',
    'route_planning',
    'obj_appearance_order',
]



def extract_uniform_frames(
    video_path: str,
    frame_dir: str,
    num_frames: int = 8,
) -> List[Tuple[int, str, float]]:
    """Extract and return uniformly sampled (frame_index, path, timestamp) tuples."""
    if num_frames <= 0:
        raise ValueError('num_frames must be greater than 0')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'VSI-Bench video not found: {video_path}')

    os.makedirs(frame_dir, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f'Invalid frame count for video: {video_path}')

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        count = min(num_frames, frame_count)
        frame_indices = np.linspace(0, frame_count - 1, count).astype(int)

        frames = []
        for frame_index in frame_indices:
            frame_index = int(frame_index)
            frame_path = os.path.join(frame_dir, f'frame_{frame_index:06d}.jpg')
            if not os.path.exists(frame_path):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                if not cv2.imwrite(
                    frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                ):
                    continue
            frames.append((frame_index, frame_path, frame_index / fps))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f'No frames extracted from video: {video_path}')

    return frames


@dataclass
class VSIBenchSample(BaseBenchmarkSample):
    options: Optional[List[str]]
    video_path: str


DEFAULT_VSIBENCH_RESULTS_ROOT = str(
    Path(__file__).resolve().parents[1] / 'results' / 'VSI-Bench'
)


class VSIBench(BaseBenchmark):
    data_specific_prompt = (
        'For multiple-choice questions, answer with the option letter only. '
        'For numerical questions, answer with the number only.'
    )

    valid_question_types = ALL_QUESTION_TYPES

    def __init__(
        self,
        data_path: str,
        question_type: List[str] = None,
        num_frames: int = 8,
        results_root: str = None,
    ):
        super().__init__()

        if question_type is None or 'all' in question_type:
            valid_types = self.valid_question_types
            invalid_types = []
        else:
            valid_types, invalid_types = [], []
            for qt in question_type:
                if qt in self.valid_question_types:
                    valid_types.append(qt)
                else:
                    invalid_types.append(qt)

        if not valid_types:
            raise ValueError(
                f'question_type {question_type} not supported. '
                f'Expected {self.valid_question_types}.'
            )
        if invalid_types:
            print(
                f'[Warning] partial question_type {invalid_types} not supported. '
                f'Expected {self.valid_question_types}.'
            )

        self.question_type = valid_types
        self.data_path = data_path
        self.num_frames = num_frames
        self.results_root = results_root or os.getenv(
            'VSIBENCH_RESULTS_ROOT', DEFAULT_VSIBENCH_RESULTS_ROOT
        )
        self.data = self.read_data()

    def read_data(self):
        jsonl_path = os.path.join(self.data_path, 'test.jsonl')
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f'VSI-Bench data file not found: {jsonl_path}')

        rows = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        data = pd.DataFrame(rows)
        data = data[data['question_type'].isin(self.question_type)].copy()

        missing = []
        for _, row in data[['dataset', 'scene_name']].drop_duplicates().iterrows():
            video_path = self._video_path(row['dataset'], row['scene_name'])
            if not os.path.exists(video_path):
                missing.append(video_path)

        print('Evaluating question types:')
        for qt in self.question_type:
            print(f'  - {qt}')
        print(f'Totally {len(data)} samples.')
        if missing:
            print(f'[Warning] {len(missing)} videos are missing. First 5:')
            for path in missing[:5]:
                print(f'  - {path}')

        return data

    def _video_path(self, dataset: str, scene_name: str) -> str:
        return os.path.join(
            self.data_path, 'videos', str(dataset), f'{scene_name}.mp4'
        )

    def _frame_dir(self, dataset: str, scene_name: str) -> str:
        return os.path.join(
            self.results_root, str(dataset), str(scene_name), 'frames'
        )

    def _ensure_frames(self, dataset: str, scene_name: str) -> List[str]:
        video_path = self._video_path(dataset, scene_name)
        frame_dir = self._frame_dir(dataset, scene_name)
        frames = extract_uniform_frames(video_path, frame_dir, self.num_frames)
        return [frame_path for _, frame_path, _ in frames]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> VSIBenchSample:
        if index >= len(self.data):
            raise IndexError(f'Index {index} out of range (0-{len(self) - 1})')

        row = self.data.iloc[index]
        options = row['options']
        question = row['question']
        if options is not None:
            options = list(options)
            question = f"{question}\nOptions:\n" + '\n'.join(options)
        else:
            options = None

        video_path = self._video_path(row['dataset'], row['scene_name'])
        images = self._ensure_frames(row['dataset'], row['scene_name'])

        return VSIBenchSample(
            sample_id=int(row['id']),
            question=question,
            question_type=row['question_type'],
            images=images,
            answer=str(row['ground_truth']),
            options=options,
            video_path=video_path,
        )

    @staticmethod
    def _fuzzy_matching(prediction: str) -> str:
        if prediction is None:
            return ''
        prediction = str(prediction).strip()
        return prediction.split(' ')[0].rstrip('.').strip()

    @classmethod
    def _extract_mca_answer(cls, prediction: str) -> Optional[str]:
        if prediction is None:
            return None

        prediction = str(prediction).strip()
        match = re.search(r'\\boxed{\s*([A-Z])\s*}', prediction, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(
            r'<\|begin_of_box\|>\s*([A-Z])\s*<\|end_of_box\|>',
            prediction,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        value = cls._fuzzy_matching(prediction)
        return value.lower() if value else None

    @classmethod
    def _extract_na_answer(cls, prediction: str) -> Optional[float]:
        if prediction is None:
            return None

        prediction = str(prediction).strip()
        match = re.search(r'\\boxed{\s*(-?\d+(?:\.\d+)?)\s*}', prediction)
        if not match:
            match = re.search(r'(-?\d+(?:\.\d+)?)', prediction)

        if not match:
            return None

        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _mean_relative_accuracy(
        prediction: Optional[float],
        target: float,
        start: float = 0.5,
        end: float = 0.95,
        interval: float = 0.05,
    ) -> float:
        if prediction is None or target is None:
            return 0.0

        num_points = int((end - start) / interval + 2)
        confidence_intervals = np.linspace(start, end, num_points)
        relative_error = abs(prediction - target) / target if target != 0 else float('inf')
        return float(np.mean(relative_error <= (1 - confidence_intervals)))

    def evaluate(
        self,
        predictions: Dict[int | str, str],
        output_dir: Optional[str] = None,
        ignore_empty: bool = False,
    ) -> Dict:
        detailed_results = []

        for _, row in self.data.iterrows():
            sample_id = int(row['id'])
            prediction = predictions.get(sample_id, '')
            if ignore_empty and (prediction is None or str(prediction).strip() == ''):
                continue

            question_type = row['question_type']
            ground_truth = str(row['ground_truth'])
            result = {
                'id': sample_id,
                'dataset': row['dataset'],
                'scene_name': row['scene_name'],
                'question_type': question_type,
                'ground_truth': ground_truth,
                'prediction': prediction,
            }

            if question_type in MCA_QUESTION_TYPES:
                extracted = self._extract_mca_answer(prediction)
                score = 1.0 if extracted == ground_truth.lower() else 0.0
                result.update({
                    'extracted_answer': extracted,
                    'score': score,
                    'metric': 'accuracy',
                })
            elif question_type in NA_QUESTION_TYPES:
                extracted = self._extract_na_answer(prediction)
                try:
                    target = float(ground_truth)
                except (TypeError, ValueError):
                    target = None
                score = self._mean_relative_accuracy(extracted, target)
                result.update({
                    'extracted_answer': extracted,
                    'score': score,
                    'metric': 'MRA:.5:.95:.05',
                })
            else:
                raise ValueError(f'Unknown question type: {question_type}')

            detailed_results.append(result)

        detail_df = pd.DataFrame(detailed_results)
        metric_averages = {}
        if not detail_df.empty:
            for question_type, group in detail_df.groupby('question_type'):
                metric = group['metric'].iloc[0]
                metric_averages[f'{question_type}_{metric}'] = float(group['score'].mean())

        direction_keys = [
            'object_rel_direction_easy_accuracy',
            'object_rel_direction_medium_accuracy',
            'object_rel_direction_hard_accuracy',
        ]
        if all(key in metric_averages for key in direction_keys):
            metric_averages['object_rel_direction_accuracy'] = float(
                np.mean([metric_averages.pop(key) for key in direction_keys])
            )

        overall = float(np.mean(list(metric_averages.values()))) if metric_averages else 0.0

        question_type_accuracy = {}
        if not detail_df.empty:
            for question_type, group in detail_df.groupby('question_type'):
                question_type_accuracy[question_type] = {
                    'score': float(group['score'].mean()),
                    'total_samples': int(len(group)),
                }

        results = {
            'total_samples': int(len(detail_df)),
            'overall': overall * 100.0,
            'question_type_metrics': {
                key: value * 100.0 for key, value in metric_averages.items()
            },
            'question_type_accuracy': question_type_accuracy,
            'detailed_results': detailed_results,
        }

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, 'results.csv')
            detail_df.to_csv(csv_path, index=False)
            self.save_results(results, os.path.join(output_dir, 'results_summary.json'))

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict, output_dir: Optional[str] = None):
        print('\n' + '=' * 64)
        print('VSI-Bench Evaluation Results')
        print('=' * 64)
        print(f'Total samples   : {results["total_samples"]:6d}')
        print(f'Overall score   : {results["overall"]:8.2f}')
        print('=' * 64)
        print('Metrics by Question Type:')
        for key, value in results['question_type_metrics'].items():
            print(f'{key:55s}: {value:7.2f}')
        print('=' * 64)
