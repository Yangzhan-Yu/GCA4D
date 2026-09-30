"""The evidence planner must survive a local thinking model.

A locally served hybrid Qwen3 emits ``<think>...</think>`` before its answer.
That is not valid JSON, so the planner used to die with
``JSONDecodeError: Expecting value: line 1 column 1``.

Run inside the gca environment:
    source scripts/gca_env.sh
    python tests/test_evidence_planner_local.py
"""

import asyncio
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.nodes.evidence_planner import QuestionEvidencePlanner  # noqa: E402

PLAN = {
    'task_type': 'counting',
    'target_entities': ['chair'],
    'reference_entities': [],
    'relation_or_metric': 'count',
    'time_constraint': None,
    'reference_frame': 'world',
    'required_evidence': ['object_track'],
    'reasoning': 'stub',
}


class _Completions:
    def __init__(self, content, reasoning=None):
        self.content = content
        self.reasoning = reasoning
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = types.SimpleNamespace(
            content=self.content, reasoning_content=self.reasoning
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )


def _client(content, reasoning=None):
    completions = _Completions(content, reasoning)
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    ), completions


def _run(content, reasoning=None, env=None):
    saved = {
        k: os.environ.get(k)
        for k in ('GCA_LLM_EXTRA_BODY', 'GCA_PLANNER_MAX_TOKENS')
    }
    try:
        for key in saved:
            os.environ.pop(key, None)
        if env:
            os.environ.update(env)
        client, completions = _client(content, reasoning)
        planner = QuestionEvidencePlanner(client=client, model='stub')
        request = asyncio.run(planner.plan(question='How many chairs?',
                                           question_type='object_counting'))
        return request, completions
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value


def main():
    failures = []
    payload = json.dumps(PLAN, ensure_ascii=False)

    # 1. plain JSON
    request, _ = _run(payload)
    if request.target_entities != ['chair']:
        failures.append(f'plain JSON parsed wrong: {request}')

    # 2. <think> block in front of the JSON (the local Qwen3 default)
    request, _ = _run(f'<think>\nLet me think about this.\n</think>\n{payload}')
    if request.target_entities != ['chair']:
        failures.append(f'<think>-prefixed answer parsed wrong: {request}')

    # 3. answer in a json code fence after thinking
    request, _ = _run(f'<think>reasoning</think>\n```json\n{payload}\n```')
    if request.target_entities != ['chair']:
        failures.append(f'fenced answer parsed wrong: {request}')

    # 4. thinking routed to reasoning_content, content empty
    request, _ = _run(None, reasoning=f'```json\n{payload}\n```')
    if request.target_entities != ['chair']:
        failures.append(f'reasoning-only answer parsed wrong: {request}')

    # 5. GCA_LLM_EXTRA_BODY must reach the request body
    _, completions = _run(
        payload,
        env={'GCA_LLM_EXTRA_BODY': '{"chat_template_kwargs": {"enable_thinking": false}}'},
    )
    sent = completions.kwargs or {}
    if sent.get('chat_template_kwargs') != {'enable_thinking': False}:
        failures.append(f'extra body not forwarded: {sent}')

    # 6. token budget override
    _, completions = _run(payload, env={'GCA_PLANNER_MAX_TOKENS': '8192'})
    if (completions.kwargs or {}).get('max_tokens') != 8192:
        failures.append(f'max_tokens override ignored: {completions.kwargs}')

    if failures:
        print('FAIL')
        for item in failures:
            print('  -', item)
        return 1
    print('PASS evidence planner with local thinking model')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
