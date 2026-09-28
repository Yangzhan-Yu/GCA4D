import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ALLOWED_IMPORT_ROOTS = {
    'collections',
    'cv2',
    'itertools',
    'json',
    'math',
    'matplotlib',
    'numpy',
    'pathlib',
    'PIL',
    'statistics',
    'typing',
    'tools.apis.generated_tool_api',
}

FORBIDDEN_CALLS = {
    'eval',
    'exec',
    '__import__',
    'compile',
    'input',
}


def _import_name(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return node.names[0].name
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module
    return ''


def _validate_ast(source: str) -> list[str]:
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f'SyntaxError: {exc}']

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = _import_name(node)
            allowed = any(
                module == item or module.startswith(item + '.')
                for item in ALLOWED_IMPORT_ROOTS
            )
            if not allowed:
                errors.append(f'Import not allowed: {module}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                errors.append(f'Call not allowed: {node.func.id}')
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            errors.append('Bare except is not allowed; catch a concrete exception')
    if '"type": "image"' in source or "'type': 'image'" in source:
        errors.append(
            'Use api.image_content(...) instead of raw {"type": "image"} content'
        )
    return errors


def _load_spec(path: Path):
    module_name = f'_generated_tool_validation_{path.stem}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'run') or not callable(module.run):
        raise RuntimeError('Generated tool must define callable run(context, **args)')
    tool_spec = getattr(module, 'TOOL_SPEC', None)
    if not isinstance(tool_spec, dict):
        raise RuntimeError('Generated tool must define TOOL_SPEC as a dict')
    return tool_spec


def validate_generated_tool(path: str | Path, timeout: int = 60) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {'valid': False, 'errors': [f'Tool file not found: {path}']}
    source = path.read_text(encoding='utf-8')
    errors = _validate_ast(source)

    command = [
        sys.executable,
        '-c',
        (
            'import json,sys;'
            'from pathlib import Path;'
            'from workflow.agentic.tool_validator import _load_spec;'
            'spec=_load_spec(Path(sys.argv[1]));'
            'print(json.dumps(spec, ensure_ascii=False))'
        ),
        str(path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        errors.append(completed.stdout.strip()[-4000:])
        return {'valid': False, 'errors': errors}
    try:
        tool_spec = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        errors.append(f'Failed to parse TOOL_SPEC: {exc}')
        return {'valid': False, 'errors': errors}

    if not tool_spec.get('name'):
        errors.append('TOOL_SPEC.name must not be empty')
    if not tool_spec.get('description'):
        errors.append('TOOL_SPEC.description must not be empty')
    if not isinstance(tool_spec.get('parameters', {}), dict):
        errors.append('TOOL_SPEC.parameters must be a JSON object')
    return {
        'valid': not errors,
        'errors': errors,
        'tool_spec': tool_spec,
        'source_path': str(path),
    }


__all__ = ['validate_generated_tool']
