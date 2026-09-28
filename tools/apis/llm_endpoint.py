"""Role-based OpenAI-compatible endpoint configuration.

The agent talks to two logically different models:

``planner``
    Text-only decisions: evidence planning, tool selection, JSON output.
    The Planner never sees pixels, so a cheap text model is usually enough.

``vlm``
    Vision calls: candidate checking (``verify_candidate``) and the optional
    VLM detection fallback.  This role must be a vision-language model.

Each role resolves its ``MODEL`` / ``BASE_URL`` / ``API_KEY`` from environment
variables, trying the role-specific prefix first and then the legacy
``AGENT_COT_REASONER_*`` single-model configuration, so existing ``API.txt``
files keep working unchanged.

Example::

    export AGENT_PLANNER_MODEL='qwen3-max'
    export AGENT_PLANNER_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
    export AGENT_PLANNER_API_KEY='sk-...'

    export AGENT_VLM_MODEL='qwen3-vl-plus'
    export AGENT_VLM_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
    export AGENT_VLM_API_KEY='sk-...'
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional


ROLE_PREFIXES: Dict[str, List[str]] = {
    'planner': ['AGENT_PLANNER', 'AGENT_COT_REASONER', 'AGENT_CODE_GENERATOR'],
    'vlm': ['AGENT_VLM', 'AGENT_COT_REASONER', 'AGENT_CODE_GENERATOR'],
}

# Some model aliases cold-start slowly (observed >90s on a first call),
# so keep a generous default and allow override via GCA_LLM_TIMEOUT.
DEFAULT_TIMEOUT = float(os.environ.get('GCA_LLM_TIMEOUT', '180'))
DEFAULT_MAX_RETRIES = int(os.environ.get('GCA_LLM_MAX_RETRIES', '2'))


class EndpointError(RuntimeError):
    pass


@dataclass
class Endpoint:
    role: str
    prefix: str
    model: str
    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES

    def describe(self) -> Dict[str, Any]:
        """A log-safe summary (never includes the api key)."""
        return {
            'role': self.role,
            'source': f'{self.prefix}_*',
            'model': self.model,
            'base_url': self.base_url,
            'api_key': f'***{self.api_key[-4:]}' if self.api_key else '',
        }


def _read_env(prefix: str) -> Dict[str, Optional[str]]:
    return {
        'model': os.environ.get(f'{prefix}_MODEL') or None,
        'base_url': os.environ.get(f'{prefix}_BASE_URL') or None,
        'api_key': os.environ.get(f'{prefix}_API_KEY') or None,
    }


def resolve_endpoint(role: str, required: bool = True) -> Optional[Endpoint]:
    """Resolve the endpoint for one role from the environment."""
    role = str(role).strip().lower()
    if role not in ROLE_PREFIXES:
        raise ValueError(f'Unknown LLM role: {role!r}; expected one of {sorted(ROLE_PREFIXES)}')

    for prefix in ROLE_PREFIXES[role]:
        values = _read_env(prefix)
        if not all(values.values()):
            continue
        return Endpoint(
            role=role,
            prefix=prefix,
            model=values['model'],
            base_url=values['base_url'],
            api_key=values['api_key'],
        )

    if not required:
        return None
    tried = ', '.join(f'{prefix}_MODEL/BASE_URL/API_KEY' for prefix in ROLE_PREFIXES[role])
    raise EndpointError(
        f'No complete configuration found for role {role!r}. '
        f'Set one of: {tried}.'
    )


def create_sync_client(endpoint: Endpoint):
    from openai import OpenAI

    return OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout,
        max_retries=endpoint.max_retries,
    )


def create_async_client(endpoint: Endpoint):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=endpoint.timeout,
        max_retries=endpoint.max_retries,
    )


# --------------------------------------------------------------------------
# response handling
# --------------------------------------------------------------------------
def extract_message_text(message) -> str:
    """Return the usable text of an assistant message.

    Handles providers that
    * return the answer in ``content`` and thinking in ``reasoning_content``;
    * put everything into ``reasoning_content`` and leave ``content`` empty;
    * inline thinking as ``<think>...</think>`` or a bare ``</think>`` marker;
    * return structured content parts instead of a plain string.
    """
    content = _as_text(getattr(message, 'content', None))
    reasoning = _as_text(getattr(message, 'reasoning_content', None))

    if content:
        lowered = content.lower()
        if '</think>' in lowered:
            _, _, tail = content.partition('</think>')
            tail = tail.strip()
            if tail:
                return tail
        return content

    if reasoning:
        # The model only produced thinking; recover the last JSON-looking block
        # if present, otherwise hand back the raw reasoning so the caller can
        # report a parse error with context.
        recovered = _last_json_block(reasoning)
        return recovered or reasoning

    raise ValueError('Model returned an empty message (no content, no reasoning_content)')


def _as_text(value) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get('text') or item.get('content') or ''))
            else:
                parts.append(str(getattr(item, 'text', '') or ''))
        return ''.join(parts).strip()
    return str(value).strip()


def _last_json_block(text: str) -> str:
    import re

    matches = re.findall(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    start = text.rfind('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return text[start:end + 1].strip()
    return ''


# --------------------------------------------------------------------------
# chat helpers with graceful parameter fallback
# --------------------------------------------------------------------------
def _is_param_error(exc: Exception, *names: str) -> bool:
    text = f'{type(exc).__name__}: {exc}'.lower()
    if not any(name.lower() in text for name in names):
        return False
    return any(token in text for token in ('unsupported', 'not supported', 'invalid', 'unexpected', 'unknown'))


def _build_kwargs(model: str, messages, max_tokens, temperature, top_p, extra=None):
    kwargs: Dict[str, Any] = {'model': model, 'messages': messages}
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens
    if temperature is not None:
        kwargs['temperature'] = temperature
    if top_p is not None:
        kwargs['top_p'] = top_p
    if extra:
        kwargs.update(extra)
    return kwargs


def _recover_kwargs(exc: Exception, model, messages, max_tokens, temperature, top_p, extra):
    """Return a degraded kwargs dict when a provider rejects optional params."""
    degraded = _build_kwargs(model, messages, max_tokens, temperature, top_p, extra)
    changed = False
    if temperature is not None and _is_param_error(exc, 'temperature'):
        degraded.pop('temperature', None)
        temperature = None
        changed = True
    if top_p is not None and _is_param_error(exc, 'top_p'):
        degraded.pop('top_p', None)
        top_p = None
        changed = True
    if max_tokens is not None and _is_param_error(exc, 'max_tokens', 'max_completion_tokens'):
        if 'max_completion_tokens' in str(exc).lower():
            degraded.pop('max_tokens', None)
            degraded['max_completion_tokens'] = max_tokens
        else:
            degraded.pop('max_tokens', None)
        changed = True
    return (degraded, changed)


async def async_chat_text(
    client,
    model: str,
    messages,
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = 0.95,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the chat completion endpoint and return plain assistant text."""
    kwargs = _build_kwargs(model, messages, max_tokens, temperature, top_p, extra)
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - providers differ
        degraded, changed = _recover_kwargs(
            exc, model, messages, max_tokens, temperature, top_p, extra
        )
        if not changed:
            raise
        response = await client.chat.completions.create(**degraded)
    return extract_message_text(response.choices[0].message)


def chat_text(
    client,
    model: str,
    messages,
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = 0.95,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Synchronous counterpart of :func:`async_chat_text`."""
    kwargs = _build_kwargs(model, messages, max_tokens, temperature, top_p, extra)
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        degraded, changed = _recover_kwargs(
            exc, model, messages, max_tokens, temperature, top_p, extra
        )
        if not changed:
            raise
        response = client.chat.completions.create(**degraded)
    return extract_message_text(response.choices[0].message)


__all__ = [
    'Endpoint',
    'EndpointError',
    'ROLE_PREFIXES',
    'resolve_endpoint',
    'create_sync_client',
    'create_async_client',
    'extract_message_text',
    'async_chat_text',
    'chat_text',
]
