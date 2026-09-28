import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ApiBudgetExceeded(RuntimeError):
    pass


_THREAD_LOCK = threading.Lock()


def _budget_path() -> Optional[Path]:
    value = os.environ.get('GCA_API_BUDGET_PATH')
    return Path(value) if value else None


def configure_api_budget(path: str | Path, max_calls: int, reset: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset or not path.exists():
        path.write_text(
            json.dumps({
                'max_calls': int(max_calls),
                'used': 0,
                'calls': [],
            }, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    os.environ['GCA_API_BUDGET_PATH'] = str(path)


def _read_budget(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {'max_calls': 0, 'used': 0, 'calls': []}
    return json.loads(path.read_text(encoding='utf-8'))


def reserve_api_call(kind: str = 'qwen', metadata: Optional[Dict[str, Any]] = None):
    path = _budget_path()
    if path is None:
        return None
    with _THREAD_LOCK:
        lock_path = path.with_suffix(path.suffix + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+') as lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            budget = _read_budget(path)
            max_calls = int(budget.get('max_calls', 0))
            used = int(budget.get('used', 0))
            if max_calls > 0 and used >= max_calls:
                raise ApiBudgetExceeded(
                    f'API budget exceeded: {used}/{max_calls} calls already used.'
                )
            used += 1
            calls = list(budget.get('calls', []))
            calls.append({
                'index': used,
                'kind': kind,
                'time': time.time(),
                'metadata': metadata or {},
            })
            budget['used'] = used
            budget['calls'] = calls[-2000:]
            temp_path = path.with_suffix(path.suffix + '.tmp')
            temp_path.write_text(
                json.dumps(budget, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            temp_path.replace(path)
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
    return used


def get_api_budget() -> Dict[str, Any]:
    path = _budget_path()
    if path is None:
        return {}
    return _read_budget(path)


__all__ = [
    'ApiBudgetExceeded',
    'configure_api_budget',
    'get_api_budget',
    'reserve_api_call',
]
