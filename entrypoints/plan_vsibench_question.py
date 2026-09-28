import argparse
import asyncio
import json
from pathlib import Path

from workflow.config import AgentConfig
from workflow.nodes.evidence_planner import QuestionEvidencePlanner
from workflow.prompts.evidence_planner import build_evidence_planner_prompt
from tools.llm_client import LLMClientFactory


def parse_args():
    parser = argparse.ArgumentParser('Plan evidence for one VSI-Bench question')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--question-id', type=int, required=True)
    parser.add_argument(
        '--data-root',
        default='/data3/Agentic-Spatial-Reasoning/gca-main/data/vsibench',
    )
    parser.add_argument('--results-root', default='/data3/Agentic-Spatial-Reasoning/gca-main/results/VSI-Bench')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', default=None)
    return parser.parse_args()


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


async def main():
    args = parse_args()
    row = load_question(
        Path(args.data_root),
        args.dataset,
        args.scene_name,
        args.question_id,
    )
    prompt = build_evidence_planner_prompt(
        question=row['question'],
        question_type=row['question_type'],
        options=row['options'],
    )

    if args.dry_run:
        print(prompt)
        return

    config = AgentConfig()
    client, model = LLMClientFactory().create_client('cot_reasoner')
    planner = QuestionEvidencePlanner(client=client, model=model)
    evidence_request = await planner.plan(
        question=row['question'],
        question_type=row['question_type'],
        options=row['options'],
    )
    output = evidence_request.to_dict()
    output.update({
        'question_id': row['id'],
        'dataset': row['dataset'],
        'scene_name': row['scene_name'],
        'question_type': row['question_type'],
        'question': row['question'],
        'options': row['options'],
        'ground_truth': row['ground_truth'],
    })

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(args.results_root)
            / args.dataset
            / args.scene_name
            / 'question_plans'
            / f'{args.question_id}.json'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f'Evidence plan saved to: {output_path}')


if __name__ == '__main__':
    asyncio.run(main())
