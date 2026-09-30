"""Static check: every name a module uses must resolve.

Three of the bugs introduced while tuning this pipeline were plain undefined
names created by a half-finished rename:

* ``cluster_positions_by_distance`` after the track clustering switched to
  ``cluster_positions_by_extent`` - the collector crashed, wrote no geometry,
  and the count came out 0;
* ``window_start`` / ``window_end`` assigned only inside one branch of
  find_bridge_frames;
* ``mask_rejections`` referenced before it was defined.

None of them are visible at import time, so the unit tests passed.  This runs
pyflakes when available and otherwise applies a small AST check, so a rename
cannot ship half-done.
"""

import ast
import builtins
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    'entrypoints/run_vsibench_agent.py',
    'entrypoints/collect_vsibench_evidence.py',
    'entrypoints/segment_with_sam3.py',
    'entrypoints/estimate_vsibench_object_size.py',
    'entrypoints/estimate_vsibench_metric_scale.py',
    'tools/apis/entity_detection_cache.py',
    'tools/utils/mask_metrics.py',
    'tools/apis/llm_endpoint.py',
    'tools/apis/evidence_sufficiency.py',
    'workflow/agentic/planner_loop.py',
    'workflow/constraints/runtime.py',
    'workflow/constraints/executor.py',
    'workflow/constraints/pipeline.py',
    'workflow/constraints/validator.py',
    'workflow/constraints/task_constraints.py',
    'workflow/constraints/evidence_profile.py',
]


MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__'}


def _bound_names(node) -> set:
    """Names bound in this scope, not descending into nested scopes."""
    names = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
                continue            # a nested scope, not this one
            if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp,
                                  ast.DictComp, ast.GeneratorExp)):
                # Own scope: its parameters and targets are bound here, and we
                # are lenient about treating them as bound in the enclosing
                # scope rather than modelling comprehension scoping exactly.
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
                    elif isinstance(sub, ast.arg):
                        names.add(sub.arg)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
            elif isinstance(child, ast.arg):
                names.add(child.arg)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            stack.append(child)
    return names


def _loaded_names(node):
    """(name, lineno) pairs loaded in this scope, not descending into nested scopes."""
    found = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue            # a nested scope, checked on its own
            if isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp,
                                  ast.DictComp, ast.GeneratorExp)):
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        found.append((sub.id, sub.lineno))
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                found.append((child.id, child.lineno))
            stack.append(child)
    return found


def _nested_functions(node):
    return [
        child for child in ast.iter_child_nodes(node)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def undefined_names(path: Path):
    """Names loaded inside a function that no enclosing scope defines."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module_names = _bound_names(tree) | MODULE_DUNDERS | set(dir(builtins))
    problems = []

    def check(func, enclosing):
        local = set(enclosing) | _bound_names(func)
        for name, lineno in _loaded_names(func):
            if name in local or name in module_names:
                continue
            problems.append((func.name, lineno, name))
        for child in _nested_functions(func):
            check(child, local)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check(node, set())
    return problems


def _pyflakes_problems(path: Path):
    exe = shutil.which('pyflakes')
    if not exe:
        return None
    result = subprocess.run(
        [exe, str(path)], capture_output=True, text=True, check=False
    )
    return result.stdout


def test_pyflakes_reports_no_undefined_names():
    checked = 0
    for rel in TARGETS:
        path = ROOT / rel
        assert path.exists(), f'{rel} is missing'
        output = _pyflakes_problems(path)
        if output is None:
            continue          # pyflakes unavailable: the AST test still runs
        bad = [
            line for line in output.splitlines()
            if 'undefined name' in line
        ]
        assert not bad, f'{rel}:\n' + '\n'.join(bad)
        checked += 1
    if checked == 0:
        print('  (pyflakes not on PATH; relying on the AST check)')


def test_ast_finds_no_undefined_names():
    failures = []
    for rel in TARGETS:
        for func, lineno, name in undefined_names(ROOT / rel):
            failures.append(f'{rel}:{lineno} in {func}(): {name}')
    assert not failures, 'undefined names:\n' + '\n'.join(failures)


def test_the_checker_actually_detects_a_half_done_rename(tmp_path=None):
    """Guard the guard: a known-bad snippet must be reported."""
    import tempfile

    source = (
        'def outer():\n'
        '    return missing_helper(1)\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sample.py'
        path.write_text(source, encoding='utf-8')
        found = undefined_names(path)
    assert any(name == 'missing_helper' for _, _, name in found), found


def _main():
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
    raise SystemExit(_main())
