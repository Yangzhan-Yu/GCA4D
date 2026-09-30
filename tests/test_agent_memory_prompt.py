"""The Planner prompt must not grow past the model's context window.

The memory is re-sent on every step and a single visibility scan can be 9k
characters.  Before this was bounded, a counting question reached 30,721 input
tokens and the local 32k-context model rejected the request outright.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.apis.agent_memory import AgentMemory  # noqa: E402
from workflow.prompts.agent_tool_planner import build_agent_tool_planner_prompt  # noqa: E402


def _fat_memory(steps=15):
    memory = AgentMemory()
    for step in range(steps):
        memory.add('planner_raw', round=0, step=step, content='x' * 3000)
        memory.add('planner_decision', round=0, step=step,
                   decision={'thought': 'y' * 300, 'tool_name': 'scan_entity_visibility'})
        memory.add(
            'tool_observation',
            round=0, step=step, tool_name='scan_entity_visibility',
            args={'entities': ['chair', 'table']},
            result={'visibility': {'chair': {'visible_frames': [{'bbox': list(range(40))}] * 60}}},
        )
    return memory


def test_prompt_summary_is_bounded():
    summary = _fat_memory().prompt_summary()
    size = len(json.dumps(summary, ensure_ascii=False))
    assert size <= 12000, f'summary too large: {size}'
    assert summary['omitted_events'] > 0
    assert summary['events'], 'summary should keep recent events'


def test_prompt_summary_drops_raw_responses():
    summary = _fat_memory()
    types = {e['event_type'] for e in summary.prompt_summary()['events']}
    assert 'planner_raw' not in types, 'raw responses duplicate planner_decision'
    assert 'tool_observation' in types


def test_prompt_summary_marks_truncation():
    memory = AgentMemory()
    memory.add('tool_observation', result={'blob': 'z' * 50000})
    events = memory.prompt_summary()['events']
    assert events and events[0].get('truncated') is True
    assert '<truncated>' in str(events[0]['data'])


def test_built_prompt_stays_under_budget_for_a_long_run():
    """Worst case: 15 steps of fat tool output, still well inside 32k tokens."""
    memory = _fat_memory(steps=15)
    tools = [
        {'name': f'tool_{i}', 'description': 'd' * 200,
         'parameters': {'type': 'object', 'properties': {'a': {'type': 'string'}}},
         'source': 'builtin'}
        for i in range(16)
    ]
    prompt = build_agent_tool_planner_prompt(
        question='How many chairs are in this room?',
        plan={'target_entities': ['chair']},
        tools=tools,
        scene_summary={'counts': {'frames': 100, 'objects': 3}},
        evidence_status={'sufficient': True},
        agent_summary=memory.prompt_summary(),
        round_index=5, max_rounds=5, step_index=15,
        allow_tool_creation=False,
    )
    # ~3.1 chars/token for this JSON-heavy prompt, measured against vLLM's
    # tokenizer.  Leave room for the 2048-token completion budget.
    estimate = len(prompt) / 3.1
    assert estimate < 24000, f'prompt would need ~{estimate:,.0f} tokens'
    assert 'truncated view' in prompt.lower()


def main():
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failures = 0
    for name, func in tests:
        try:
            func()
            print(f'PASS {name}')
        except Exception:
            failures += 1
            print(f'FAIL {name}')
            traceback.print_exc()
    print(f'\n{len(tests) - failures}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
