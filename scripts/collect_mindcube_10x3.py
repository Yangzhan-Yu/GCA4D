#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT = Path('/data3/Agentic-Spatial-Reasoning/gca-main')
RUN_ROOT = ROOT / 'work_dir/mindcube_10x3_qwen3vl235b'
TYPES = ['rotation', 'around', 'among']


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    text = str(text).strip()
    for pattern in [
        r'\\boxed{\s*([A-E])\s*}',
        r'<\|begin_of_box\|>\s*([A-E])\s*<\|end_of_box\|>',
        r'\b([A-E])\.',
        r'\b([A-E])\b',
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


all_rows = []
summary = {}
for question_type in TYPES:
    prediction_file = RUN_ROOT / question_type / 'predictions.jsonl'
    if not prediction_file.exists():
        print(f'Missing predictions: {prediction_file}')
        continue

    rows = []
    for line in prediction_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prediction = row.get('content', '')
        ground_truth = row.get('ground_truth')
        extracted = extract_answer(prediction)
        is_correct = extracted == ground_truth
        review = {
            'sample_id': row.get('sample_id'),
            'question_type': question_type,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'extracted_answer': extracted,
            'is_correct': is_correct,
            'question': row.get('question', ''),
        }
        rows.append(review)
        all_rows.append(review)

    correct = sum(row['is_correct'] for row in rows)
    total = len(rows)
    summary[question_type] = {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total else 0.0,
    }

if all_rows:
    correct = sum(row['is_correct'] for row in all_rows)
    total = len(all_rows)
    summary['overall'] = {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total else 0.0,
    }

review_path = RUN_ROOT / 'results_review.csv'
with review_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'sample_id', 'question_type', 'ground_truth', 'prediction',
        'extracted_answer', 'is_correct', 'question'
    ])
    writer.writeheader()
    writer.writerows(all_rows)

summary_path = RUN_ROOT / 'results_summary.json'
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f'Review saved to: {review_path}')
print(f'Summary saved to: {summary_path}')
