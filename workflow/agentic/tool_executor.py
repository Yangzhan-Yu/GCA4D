import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from tools.apis.api_budget import ApiBudgetExceeded
from workflow.agentic.tool_registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_name: str, args: Dict[str, Any], context: Dict[str, Any]):
        spec = self.registry.get(tool_name)
        if spec.source_path:
            return self._execute_generated(spec, args or {}, context)
        try:
            result = spec.handler(context=context, **(args or {}))
        except ApiBudgetExceeded as exc:
            return {'error': str(exc), 'budget_exceeded': True}
        except Exception as exc:
            return {
                'error': f'{type(exc).__name__}: {exc}',
                'tool_execution_error': True,
            }
        if not isinstance(result, dict):
            return {'result': result}
        return result

    def _execute_generated(self, spec, args: Dict[str, Any], context: Dict[str, Any]):
        scene_root = Path(context['scene_root'])
        run_dir = scene_root / 'generated_tools' / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"run_{int(time.time() * 1000)}"
        input_path = run_dir / f'{stem}_input.json'
        output_path = run_dir / f'{stem}_output.json'
        safe_context = {
            'dataset': context['dataset'],
            'scene_name': context['scene_name'],
            'question_id': context['question_id'],
            'data_root': str(context['data_root']),
            'results_root': str(context['results_root']),
            'scene_root': str(context['scene_root']),
            'device': context.get('device', 'cuda'),
            'model': context['model'],
        }
        input_path.write_text(
            json.dumps({'context': safe_context, 'args': args}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        command = [
            sys.executable,
            '-m',
            'workflow.agentic.generated_tool_runner',
            '--module',
            Path(spec.source_path).stem,
            '--input-json',
            str(input_path),
            '--output-json',
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                'error': f'Generated tool {spec.name} timed out',
                'output': (exc.stdout or '')[-10000:],
            }
        output = completed.stdout[-10000:]
        if completed.returncode != 0:
            return {
                'error': f'Generated tool {spec.name} failed',
                'returncode': completed.returncode,
                'output': output,
            }
        if not output_path.exists():
            return {
                'error': f'Generated tool {spec.name} produced no output',
                'returncode': completed.returncode,
                'output': output,
            }
        try:
            result = json.loads(output_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            return {
                'error': f'Generated tool {spec.name} returned invalid JSON: {exc}',
                'output': output,
            }
        result.setdefault('generated_tool', spec.name)
        result.setdefault('subprocess_output', output[-2000:])
        return result
