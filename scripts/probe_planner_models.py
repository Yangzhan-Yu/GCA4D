#!/usr/bin/env python
"""Probe candidate Planner models against the real agent prompt.

The Planner only needs to read a JSON state and emit one JSON decision, so the
right question is not "which model is smartest" but "which model reliably
returns the expected schema, quickly and cheaply".

Usage::

    source scripts/gca_env.sh
    python scripts/probe_planner_models.py \
        --models qwen3.5-flash qwen3.5-plus qwen3-max deepseek-v4-flash
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.apis.llm_endpoint import (  # noqa: E402
    async_chat_text,
    create_async_client,
    resolve_endpoint,
)
from workflow.agentic.tool_registry import ToolRegistry  # noqa: E402
from workflow.prompts.agent_tool_planner import build_agent_tool_planner_prompt  # noqa: E402
from workflow.utils.parse_utils import parse_first_json_object  # noqa: E402


DEFAULT_MODELS = [
    'qwen3.5-flash',
    'qwen3.7-flash',
    'qwen3.5-plus',
    'qwen3-max',
    'deepseek-v4-flash',
    'ZHIPU/GLM-5.3-Flash',
]

QUESTION = 'What is the distance between the sofa and the stove?'
PLAN = {
    'question_id': 680,
    'question_type': 'object_abs_distance',
    'question': QUESTION,
    'target_entities': ['sofa', 'stove'],
    'reference_entities': [],
    'reference_frame': 'world',
    'relation_or_metric': 'closest surface distance in meters',
    'required_evidence': ['3d_points', 'metric_scale', 'coordinate_frame'],
    'options': [],
}
CONSTRAINT = {
    'operation': {'operation': 'surface_distance', 'unit': 'm', 'options': []},
    'entities': [
        {'role': 'entity_a', 'category': 'sofa'},
        {'role': 'entity_b', 'category': 'stove'},
    ],
    'reference_frame': {'frame_type': 'world'},
}
SCENE_SUMMARY = {'counts': {'frames': 22, 'objects': 2, 'evidence_candidate_frames': 6}}
EVIDENCE_STATUS = {'sufficient': True, 'entities': {'sofa': 2, 'stove': 2}}
TOOLS = [
    {'name': 'check_evidence', 'description': 'Read the current evidence status.', 'parameters': {'type': 'object', 'properties': {}}, 'source': 'builtin'},
    {'name': 'execute_operation', 'description': 'Run the fixed geometric operation and store a verified result.', 'parameters': {'type': 'object', 'properties': {}}, 'source': 'builtin'},
    {'name': 'get_task_constraint', 'description': 'Show the compiled task constraint and validation.', 'parameters': {'type': 'object', 'properties': {}}, 'source': 'builtin'},
    {'name': 'collect_question_evidence', 'description': 'Detect/segment/reconstruct question evidence.', 'parameters': {'type': 'object', 'properties': {}}, 'source': 'builtin'},
]


def build_prompt():
    return build_agent_tool_planner_prompt(
        question=QUESTION,
        plan=PLAN,
        tools=TOOLS,
        scene_summary=SCENE_SUMMARY,
        evidence_status=EVIDENCE_STATUS,
        agent_summary={},
        round_index=1,
        max_rounds=5,
        step_index=1,
        max_steps_per_round=None,
        allow_tool_creation=False,
        task_constraint=CONSTRAINT,
        constraint_state={
            'bindings': {'entity_a': 'sofa_00', 'entity_b': 'stove_00'},
            'validation': {'valid': True},
            'operation_results': {},
            'verified_operation_results': {},
        },
    )


async def probe(model, client, prompt, max_tokens):
    started = time.perf_counter()
    record = {'model': model}
    try:
        text = await async_chat_text(
            client, model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens, temperature=0.0, top_p=0.95,
        )
        record['latency_s'] = round(time.perf_counter() - started, 2)
        record['text'] = text
        try:
            decision = parse_first_json_object(text)
            record['json_ok'] = True
            record['decision'] = decision
        except Exception as exc:  # noqa: BLE001
            record['json_ok'] = False
            record['error'] = f'JSON: {exc}'
    except Exception as exc:  # noqa: BLE001
        record['latency_s'] = round(time.perf_counter() - started, 2)
        record['json_ok'] = False
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=DEFAULT_MODELS)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args()

    endpoint = resolve_endpoint('planner')
    print(f'Using {endpoint.describe()["source"]} for base_url={endpoint.base_url}')
    print(f'Probing {len(args.models)} model(s), {args.repeat} run(s) each\n')

    import asyncio

    prompt = build_prompt()
    print(f'Planner prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)\n')

    client = create_async_client(endpoint)

    async def run_all():
        results = []
        for model in args.models:
            for index in range(args.repeat):
                record = await probe(model, client, prompt, args.max_tokens)
                record['run'] = index + 1
                results.append(record)
                status = 'OK  ' if record.get('json_ok') else 'FAIL'
                detail = ''
                if record.get('json_ok'):
                    decision = record['decision']
                    keys = sorted(decision.keys())
                    detail = f'keys={keys}'
                else:
                    detail = str(record.get('error'))[:110]
                print(
                    f'[{status}] {model:26s} {record["latency_s"]:6.2f}s  {detail}',
                    flush=True,
                )
        return results

    results = asyncio.run(run_all())

    print('\n--- first valid decision per model ---')
    for model in args.models:
        hit = next(
            (r for r in results if r['model'] == model and r.get('json_ok')),
            None,
        )
        if hit:
            print(f'{model}: {json.dumps(hit["decision"], ensure_ascii=False)}')
        else:
            print(f'{model}: NO VALID JSON')

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                [{k: v for k, v in r.items() if k != "text"} for r in results],
                ensure_ascii=False, indent=2,
            ) + '\n',
            encoding='utf-8',
        )
        print(f'\nWrote {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
