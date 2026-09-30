"""Tests for role-based LLM endpoint resolution and response handling."""

import asyncio
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.apis.llm_endpoint import (  # noqa: E402
    ROLE_PREFIXES,
    _recover_kwargs,
    planner_max_tokens,
    request_extra,
    async_chat_text,
    chat_text,
    extract_message_text,
    resolve_endpoint,
)

_ROLE_VARS = ('AGENT_PLANNER', 'AGENT_VLM', 'AGENT_COT_REASONER', 'AGENT_CODE_GENERATOR')


def _clear():
    for key in list(os.environ):
        if key.startswith(_ROLE_VARS):
            del os.environ[key]


def _legacy():
    os.environ.update({
        'AGENT_COT_REASONER_MODEL': 'legacy-model',
        'AGENT_COT_REASONER_BASE_URL': 'https://legacy/v1',
        'AGENT_COT_REASONER_API_KEY': 'sk-legacy-1234',
    })


def test_legacy_config_serves_planner():
    _clear()
    _legacy()
    try:
        endpoint = resolve_endpoint('planner')
        assert endpoint.model == 'legacy-model'
        assert endpoint.prefix == 'AGENT_COT_REASONER'
    finally:
        _clear()


def test_role_specific_overrides_legacy():
    _clear()
    _legacy()
    os.environ.update({
        'AGENT_PLANNER_MODEL': 'cheap-text-model',
        'AGENT_PLANNER_BASE_URL': 'https://planner/v1',
        'AGENT_PLANNER_API_KEY': 'sk-planner-9999',
    })
    try:
        assert resolve_endpoint('planner').model == 'cheap-text-model'
        assert resolve_endpoint('planner').prefix == 'AGENT_PLANNER'
    finally:
        _clear()


def test_request_extra_reads_env_json():
    saved = os.environ.pop('GCA_LLM_EXTRA_BODY', None)
    try:
        assert request_extra() == {}
        os.environ['GCA_LLM_EXTRA_BODY'] = (
            '{"chat_template_kwargs": {"enable_thinking": false}}'
        )
        assert request_extra() == {
            'chat_template_kwargs': {'enable_thinking': False}
        }
        os.environ['GCA_LLM_EXTRA_BODY'] = 'not json'
        try:
            request_extra()
        except ValueError as exc:
            assert 'not valid JSON' in str(exc)
        else:
            raise AssertionError('bad JSON should raise')
    finally:
        os.environ.pop('GCA_LLM_EXTRA_BODY', None)
        if saved is not None:
            os.environ['GCA_LLM_EXTRA_BODY'] = saved


def test_planner_max_tokens_default_and_override():
    saved = os.environ.pop('GCA_PLANNER_MAX_TOKENS', None)
    try:
        assert planner_max_tokens(2048) == 2048
        os.environ['GCA_PLANNER_MAX_TOKENS'] = '8192'
        assert planner_max_tokens(2048) == 8192
        os.environ['GCA_PLANNER_MAX_TOKENS'] = '1'
        assert planner_max_tokens(2048) == 64
    finally:
        os.environ.pop('GCA_PLANNER_MAX_TOKENS', None)
        if saved is not None:
            os.environ['GCA_PLANNER_MAX_TOKENS'] = saved


def test_unsupported_extra_key_is_dropped():
    """A cloud endpoint with no chat_template_kwargs must still work."""
    exc = Exception("BadRequestError: Unsupported parameter: chat_template_kwargs")
    kwargs, changed = _recover_kwargs(
        exc, 'm', [{'role': 'user', 'content': 'x'}], 100, 0.0, 0.95,
        {'chat_template_kwargs': {'enable_thinking': False}},
    )
    assert changed
    assert 'chat_template_kwargs' not in kwargs


def test_vlm_role_is_removed():
    """The agent is planner-only; there must be no vision role."""
    _clear()
    _legacy()
    os.environ.update({
        'AGENT_VLM_MODEL': 'some-vl-model',
        'AGENT_VLM_BASE_URL': 'https://vlm/v1',
        'AGENT_VLM_API_KEY': 'sk-vlm-0000',
    })
    try:
        assert 'vlm' not in ROLE_PREFIXES
        try:
            resolve_endpoint('vlm')
        except ValueError as exc:
            assert 'planner-only' in str(exc)
        else:
            raise AssertionError('the vlm role should no longer resolve')
    finally:
        _clear()


def test_incomplete_role_config_falls_back():
    _clear()
    _legacy()
    os.environ.update({
        'AGENT_PLANNER_MODEL': 'half-configured',
        'AGENT_PLANNER_BASE_URL': 'https://planner/v1',
        # no API key on purpose
    })
    try:
        assert resolve_endpoint('planner').model == 'legacy-model'
    finally:
        _clear()


def test_describe_never_leaks_key():
    _clear()
    _legacy()
    try:
        described = resolve_endpoint('planner').describe()
        assert described['api_key'] == '***1234'
        assert 'sk-legacy-1234' not in str(described)
    finally:
        _clear()


def test_extract_message_text_variants():
    make = lambda c=None, r=None: types.SimpleNamespace(content=c, reasoning_content=r)
    assert extract_message_text(make(c='plain')) == 'plain'
    assert extract_message_text(make(c='<think>x</think>{"done":true}')) == '{"done":true}'
    assert extract_message_text(
        make(r='thinking\n```json\n{"done":false}\n```')
    ) == '{"done":false}'
    assert extract_message_text(make(c=[{'type': 'text', 'text': 'part'}])) == 'part'
    try:
        extract_message_text(make())
    except ValueError:
        pass
    else:
        raise AssertionError('empty message should raise')


def test_recover_kwargs_handles_unknown_temperature():
    exc = Exception('BadRequestError: Unsupported parameter: temperature')
    kwargs, changed = _recover_kwargs(
        exc, 'm', [{'role': 'user', 'content': 'x'}], 100, 0.0, 0.95, None
    )
    assert changed and 'temperature' not in kwargs


def test_recover_kwargs_switches_max_tokens_field():
    exc = Exception(
        "BadRequestError: Unsupported parameter: 'max_tokens' is not supported. "
        "Use 'max_completion_tokens' instead."
    )
    kwargs, changed = _recover_kwargs(exc, 'm', [], 100, 0.0, 0.95, None)
    assert changed
    assert 'max_tokens' not in kwargs
    assert kwargs['max_completion_tokens'] == 100


class _FakeCompletions:
    def __init__(self, fail_first=None):
        self.calls = []
        self.fail_first = fail_first

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first and len(self.calls) == 1:
            raise self.fail_first
        msg = types.SimpleNamespace(content='ok', reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeAsyncCompletions(_FakeCompletions):
    async def create(self, **kwargs):
        return super().create(**kwargs)


def _fake_client(completions):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )


def test_chat_text_retries_without_temperature():
    completions = _FakeCompletions(
        fail_first=Exception('Unsupported parameter: temperature')
    )
    text = chat_text(_fake_client(completions), 'm', [{'role': 'user', 'content': 'x'}])
    assert text == 'ok'
    assert len(completions.calls) == 2
    assert 'temperature' not in completions.calls[1]
    assert 'temperature' in completions.calls[0]


def test_async_chat_text_retries_without_temperature():
    completions = _FakeAsyncCompletions(
        fail_first=Exception('Unsupported parameter: temperature')
    )
    text = asyncio.run(
        async_chat_text(_fake_client(completions), 'm', [{'role': 'user', 'content': 'x'}])
    )
    assert text == 'ok'
    assert len(completions.calls) == 2


def _main():
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith('test_') and callable(obj)
    ]
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
    raise SystemExit(_main())
