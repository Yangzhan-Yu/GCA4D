"""Run ``run_vsibench_agent.main()`` with the LLM calls stubbed.

The other tests import the entrypoint module and exercise individual helpers,
which does not catch refactor mistakes *inside* ``main()`` (e.g. a variable
that was renamed in one place but not another).  This test drives ``main()``
through its whole wiring with the network-dependent pieces replaced by stubs,
using a temporary results root.

Run inside the gca environment::

    source scripts/gca_env.sh
    python tests/test_entrypoint_smoke.py
"""

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import entrypoints.run_vsibench_agent as agent  # noqa: E402


QUESTION = {
    'id': 1,
    'dataset': 'arkitscenes',
    'scene_name': '41069025',
    'question_type': 'object_counting',
    'question': 'How many chair(s) are in this room?',
    'options': None,
    'ground_truth': '2',
}

FAKE_PLAN = {
    'task_type': 'counting',
    'target_entities': ['chair'],
    'reference_entities': [],
    'relation_or_metric': 'count of chairs',
    'time_constraint': None,
    'reference_frame': 'world',
    'required_evidence': ['object_track'],
    'reasoning': 'stub',
}


class _StubEvidenceRequest:
    def to_dict(self):
        return dict(FAKE_PLAN)


class _StubEvidencePlanner:
    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model

    async def plan(self, question, question_type, options):
        return _StubEvidenceRequest()


class _StubPlannerLoop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # main() must hand the loop a resolved planner model, not a raw name
        assert kwargs.get('model'), 'PlannerLoop received no model'

    async def run(self, question, plan):
        return {'done': True, 'final_answer': '2', 'rounds': 1, 'steps': 3}


def _check_tools_never_return_none():
    """No agent tool may fall through and return None.

    Regression: after the VLM counting path was removed,
    ``count_entities_in_video`` lost its trailing return.  With insufficient
    evidence it returned None, the Planner read that as a tool failure and then
    chased metric scale for a counting question until it hit the step cap.
    """
    import ast
    import inspect

    checks = []
    source = inspect.getsource(agent)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(isinstance(n, ast.Return) for n in ast.walk(node)):
            continue          # helper with no return value (e.g. a mutator)
        # Return and Raise both terminate; anything else can fall out of the
        # function and hand the Planner a bare None.
        if not isinstance(node.body[-1], (ast.Return, ast.Raise)):
            checks.append(
                f'{node.name} has returns but its last statement is '
                f'{type(node.body[-1]).__name__}; it can fall through as None'
            )
    return checks


def _check_repeat_guard():
    """A third identical tool call must be refused, not executed again.

    Observed: the local Planner called collect_question_evidence four times
    with byte-identical arguments and burned its whole step budget.
    """
    import asyncio
    import json as _json

    from workflow.agentic.planner_loop import PlannerLoop
    from workflow.agentic.tool_registry import ToolRegistry, ToolSpec

    calls = []

    def ping(context, **kwargs):
        calls.append(kwargs)
        return {'ok': True}

    registry = ToolRegistry()
    registry.register(ToolSpec(name='ping', description='d', parameters={}, handler=ping))

    class _Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise AssertionError('loop should not reach the third call')

    decisions = [
        {'thought': 't', 'done': False, 'mode': 'call_tool', 'tool_name': 'ping', 'args': {'a': 1}},
        {'thought': 't', 'done': False, 'mode': 'call_tool', 'tool_name': 'ping', 'args': {'a': 1}},
        {'thought': 't', 'done': False, 'mode': 'call_tool', 'tool_name': 'ping', 'args': {'a': 1}},
        {'thought': 't', 'done': True, 'final_answer': 'done'},
    ]

    async def fake_chat_text(client, model, messages, **kwargs):
        return _json.dumps(decisions.pop(0))

    import unittest.mock as mock
    import workflow.agentic.planner_loop as planner_loop_module

    loop = PlannerLoop(
        client=_Client(), model='stub', registry=registry, context={},
        max_rounds=1, max_total_steps=10,
    )
    # planner_loop imported the name directly, so patch it there
    with mock.patch.object(
        planner_loop_module, 'async_chat_text', fake_chat_text
    ):
        asyncio.run(loop.run(question='q', plan={}))

    return [] if len(calls) == 2 else [
        f'expected the third identical call to be refused, handler ran '
        f'{len(calls)} time(s)'
    ]


def _check_unparseable_planner_output():
    """A malformed planner reply must end the run gracefully, not crash.

    Observed: the local planner emitted an unclosed  thinking block that used
    the whole token budget.  parse_first_json_object raised, the exception
    propagated out of main(), and no agent_result.json was written at all -
    so the official evaluator had no prediction to score.
    """
    import asyncio
    import json as _json
    import unittest.mock as mock

    import workflow.agentic.planner_loop as planner_loop_module
    from workflow.agentic.planner_loop import PlannerLoop
    from workflow.agentic.tool_registry import ToolRegistry, ToolSpec

    registry = ToolRegistry()
    registry.register(ToolSpec(name='ping', description='d', parameters={},
                               handler=lambda context, **kw: {'ok': True}))

    class _Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise AssertionError('unreachable')

    replies = ['<think>thinking and never closing the block' + 'x' * 500] * 4

    async def fake_chat_text(client, model, messages, **kwargs):
        return replies.pop(0)

    loop = PlannerLoop(
        client=_Client(), model='stub', registry=registry, context={},
        max_rounds=1, max_total_steps=10,
    )
    with mock.patch.object(planner_loop_module, 'async_chat_text', fake_chat_text):
        result = asyncio.run(loop.run(question='q', plan={}))

    checks = []
    if result.get('done') is not False:
        checks.append(f'expected done=False, got {result!r}')
    if 'not a valid' not in str(result.get('error', '')):
        checks.append(f'error should explain the parse failure: {result.get("error")!r}')
    return checks


def _check_plan_completeness_repair():
    """Entities the question needs must end up in the evidence plan.

    Regression: "standing by the stove and facing the sofa, is the tv to the
    left or right?" was planned as target=[tv], reference=[sofa].  The stove
    was never collected, so the constraint's origin role could never bind and
    the run looped for 15 steps.
    """
    from workflow.constraints.task_constraints import compile_task_constraint

    plan = {
        'question_id': 1236,
        'question_type': 'object_rel_direction_easy',
        'question': ('If I am standing by the stove and facing the sofa, is the '
                     'tv to the left or the right of the sofa?'),
        'target_entities': ['tv'],
        'reference_entities': ['sofa'],
        'options': ['A. left', 'B. right'],
    }
    constraint = compile_task_constraint(plan)
    needed = {e.category for e in constraint.entities}
    listed = set(plan['target_entities']) | set(plan['reference_entities'])

    checks = []
    if 'stove' not in needed:
        checks.append(f'the constraint should need the stove: {needed}')
    if 'stove' in listed:
        checks.append('precondition: the sample plan should be missing stove')
    # the repair the entrypoint performs
    plan['reference_entities'] = list(
        dict.fromkeys(plan['reference_entities'] + sorted(needed - listed))
    )
    if 'stove' not in plan['reference_entities']:
        checks.append('repair did not add the missing entity')
    if set(plan['target_entities']) | set(plan['reference_entities']) != needed:
        checks.append('repaired plan still does not cover the constraint')
    return checks


def _check_progress_watchdog():
    """Ending must say whether the budget or the reasoning was the problem.

    On free local inference the step budget must never be what stops a run
    that was still improving.  The progress score rises when a role gets
    bound, an operation runs, or a result is verified.
    """
    from workflow.agentic.planner_loop import PlannerLoop
    from workflow.agentic.tool_registry import ToolRegistry

    loop = PlannerLoop(
        client=None, model='stub', registry=ToolRegistry(), context={},
        max_total_steps=30,
    )
    score = loop._progress_score

    checks = []
    idle = {'constraint_validation': {'valid': False, 'resolved_bindings': {}}}
    bound = {'constraint_validation': {'valid': True, 'resolved_bindings': {'a': 'x'}}}
    executed = dict(bound, operation_results={'op_1': {}})
    verified = dict(executed, verified_operation_results={'op_1': {}})

    if not (score(idle) < score(bound) < score(executed) < score(verified)):
        checks.append(
            'progress score should rise with bound roles, an operation '
            f'result and a verified result: {[score(x) for x in (idle, bound, executed, verified)]}'
        )

    # flat state -> counter climbs; any improvement resets it
    loop._track_progress(idle)
    loop._track_progress(idle)
    loop._track_progress(idle)
    if loop._steps_since_progress != 2:
        checks.append(f'flat steps should accumulate: {loop._steps_since_progress}')
    loop._track_progress(bound)
    if loop._steps_since_progress != 0:
        checks.append('an improvement should reset the counter')

    # the watchdog must not fire before a reasonable window
    if loop.max_steps_without_progress < 8:
        checks.append(f'window too short: {loop.max_steps_without_progress}')
    return checks


def _check_budget_pressure_nudge():
    """A wandering Planner must be pushed to execute_operation before the cap.

    Observed on the direction questions: three point clouds already existed,
    but the run ended at the step cap with no operation result because the
    Planner kept re-checking detections.
    """
    import json as _json

    from workflow.agentic.planner_loop import PlannerLoop
    from workflow.agentic.tool_registry import ToolRegistry

    loop = PlannerLoop(
        client=None, model='stub', registry=ToolRegistry(), context={},
        max_total_steps=10,
    )
    # The loop's state_provider summarises the constraint with operation as a
    # STRING.  The compiled constraint uses a nested dict.  A mismatch here
    # crashed every question at step 1 with AttributeError, so both shapes are
    # exercised.
    state_shape = {'operation': 'surface_distance', 'unit': 'm'}
    compiled_shape = {'operation': {'operation': 'surface_distance'}}

    checks = []
    for name, constraint in (('state provider', state_shape),
                             ('compiled', compiled_shape)):
        early = loop._budget_pressure_note({'task_constraint': constraint}, 2)
        if early:
            checks.append(f'[{name}] should not nudge at 2/10: {early!r}')

        late = loop._budget_pressure_note({'task_constraint': constraint}, 7)
        if 'execute_operation' not in late or 'surface_distance' not in late:
            checks.append(f'[{name}] nudge should name the operation: {late!r}')

        done = loop._budget_pressure_note(
            {'task_constraint': constraint,
             'verified_operation_results': {'surface_distance_01': {}}},
            7,
        )
        if 'Finalize NOW' not in done:
            checks.append(f'[{name}] should tell it to finalize: {done!r}')

    # no executable operation (e.g. an unsupported type): stay quiet
    for empty in ({'operation': None}, {'operation': {'operation': None}}, {}):
        quiet = loop._budget_pressure_note({'task_constraint': empty}, 9)
        if quiet:
            checks.append(f'should not nudge for {empty}: {quiet!r}')

    # and the safe wrapper must never raise, whatever it is handed
    for broken in ({'task_constraint': 'oops'},
                   {'task_constraint': {'operation': 123}},
                   {}):
        try:
            loop._budget_pressure_note_safe(broken, 9)
        except Exception as exc:  # noqa: BLE001
            checks.append(f'the safe wrapper raised on {broken}: {exc!r}')

    return checks


def _check_sanitize_plan_drops_frame_vocabulary():
    """A reference FRAME value must not be treated as a reference ENTITY.

    Regression: a local model emitted reference_frame='world' and also put the
    literal 'world' into reference_entities, so the evidence stage spent a
    SAM3/DINO pass per frame trying to detect an object called "world".
    """
    checks = []
    sanitize = agent.sanitize_plan

    got = sanitize({
        'reference_frame': 'world',
        'reference_entities': ['world', 'sofa'],
        'target_entities': ['stove'],
    })
    if got['reference_entities'] != ['sofa']:
        checks.append(f"'world' should be dropped: {got['reference_entities']}")

    # an object: anchor IS a real object and must stay bindable
    got = sanitize({
        'reference_frame': 'object:sofa',
        'reference_entities': ['sofa', 'stove'],
        'target_entities': ['tv'],
    })
    if got['reference_entities'] != ['sofa', 'stove']:
        checks.append(f'object anchor must survive: {got["reference_entities"]}')

    got = sanitize({
        'reference_frame': 'camera:003360',
        'reference_entities': ['camera:003360'],
        'target_entities': ['tv'],
    })
    if got['reference_entities'] != []:
        checks.append(f'camera anchor is not an object: {got["reference_entities"]}')

    got = sanitize({
        'reference_frame': 'world',
        'reference_entities': ['room'],
        'target_entities': ['stove'],
    })
    if got['reference_entities'] != []:
        checks.append(f'room is not bindable: {got["reference_entities"]}')

    # a target list is only filtered when something real remains
    got = sanitize({
        'reference_frame': 'world',
        'reference_entities': [],
        'target_entities': ['world'],
    })
    if got['target_entities'] != ['world']:
        checks.append(
            'should not empty the target list entirely: '
            f'{got["target_entities"]}'
        )
    return checks


def _check_bind_normalises_geometry_less_instances():
    """Binding a track without geometry must be auto-corrected.

    query_tracks returns observation sub-tracks (tv_01 ...) with no point cloud
    of their own; only <category>_00 is written.  Validation reported the
    problem but the Planner responded by re-scanning for a dozen steps instead
    of rebinding, so the tool corrects it.
    """
    import tempfile
    from unittest import mock

    import numpy as np

    from tools.apis.four_d_memory import MemoryObject, open_vsibench_memory
    from workflow.agentic.tool_registry import ToolRegistry
    from workflow.constraints.pipeline import save_constraint
    from workflow.constraints.task_constraints import compile_task_constraint

    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene_root = root / 'arkitscenes' / 'scene1'
        scene_root.mkdir(parents=True)
        store = open_vsibench_memory(
            data_root=root, dataset='arkitscenes', scene_name='scene1',
            scene_root=scene_root,
        )
        geometry = store.root_dir / 'geometry'
        geometry.mkdir(parents=True, exist_ok=True)
        for object_id in ('tv_00', 'sofa_00', 'stove_00'):
            np.savez_compressed(
                geometry / f'{object_id}_points.npz',
                points=np.zeros((10, 3), dtype='float32'),
            )
            store.upsert_object(MemoryObject(
                object_id=object_id, category=object_id.rsplit('_', 1)[0]
            ))
        plan = {
            'question_id': 957,
            'question_type': 'object_rel_direction_hard',
            'question': 'q',
            'target_entities': ['tv'],
            'reference_entities': ['stove', 'sofa'],
            'options': ['A. front-left', 'B. back-right'],
        }
        constraint = compile_task_constraint(plan).to_dict()
        save_constraint(constraint, scene_root, 957)
        registry = agent.register_tools(ToolRegistry())
        context = {
            'store': store, 'scene_root': scene_root, 'question_id': 957,
            'dataset': 'arkitscenes', 'scene_name': 'scene1', 'device': 'cpu',
            'data_root': root, 'results_root': root,
            'task_constraint': constraint,
        }
        out = registry.get('bind_constraint_entities').handler(
            context=context,
            bindings={'origin': 'stove_00', 'forward': 'sofa_00',
                      'target': 'tv_01'},
        )
        if out['bindings'].get('target') != 'tv_00':
            checks.append(f'tv_01 should become tv_00: {out["bindings"]}')
        if 'target' not in (out.get('corrected') or {}):
            checks.append(f'the correction should be reported: {out}')
        if out['bindings'].get('origin') != 'stove_00':
            checks.append('valid ids must be left alone')
    return checks


def _check_find_bridge_frames_runs():
    """Actually call the bridge tool on both code paths.

    Regression: the interval-sampling branch never assigned window_start /
    window_end, so the tool raised UnboundLocalError at its return statement.
    The Planner saw a failing tool, wandered for 13 steps and the run ended
    with no answer at all.  Nothing short of calling the tool catches this.
    """
    import json as _json
    import tempfile

    import cv2
    import numpy as np

    from tools.apis.four_d_memory import FrameRecord, open_vsibench_memory
    from workflow.agentic.tool_registry import ToolRegistry

    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_root = root / 'data'
        video_dir = data_root / 'videos' / 'arkitscenes'
        video_dir.mkdir(parents=True)
        video_path = video_dir / 'scene1.mp4'
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (32, 32)
        )
        for _ in range(40):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

        scene_root = root / 'results' / 'arkitscenes' / 'scene1'
        scene_root.mkdir(parents=True)
        store = open_vsibench_memory(
            data_root=data_root, dataset='arkitscenes',
            scene_name='scene1', scene_root=scene_root,
        )
        for index in range(4):
            store.add_frame(FrameRecord(
                frame_id=f'{index * 10:06d}', timestamp=float(index),
                dataset='arkitscenes', scene_name='scene1',
                video_path=str(video_path), frame_path=str(video_path),
                metadata={'role': 'visibility_scan'},
            ))

        evidence = scene_root / 'evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        registry = agent.register_tools(ToolRegistry())
        tool = registry.get('find_bridge_frames').handler
        context = {
            'store': store, 'scene_root': scene_root, 'dataset': 'arkitscenes',
            'scene_name': 'scene1', 'data_root': data_root,
            'evidence_profile': {'bridge_interval_seconds': 1.0,
                                 'max_bridge_frames': 16,
                                 'bridge_padding_seconds': 0.0},
        }

        # Branch 1: intervals present -> interval sampling
        (evidence / 'visibility_scan.json').write_text(_json.dumps({
            'visibility': {'chair': {
                'first_seen': 0.0, 'last_seen': 3.0,
                'intervals': [[0.0, 1.0], [3.0, 3.0]],
            }},
            'suggested_bridge_window': {
                'start_time': 0.0, 'end_time': 3.0,
                'per_entity_interval': {'chair': [0.0, 3.0]},
            },
        }), encoding='utf-8')
        out = tool(context=context, anchor_frame_ids=[])
        if out.get('error'):
            checks.append(f'interval branch failed: {out["error"]}')
        else:
            for key in ('window_start', 'window_end', 'bridge_frame_ids'):
                if key not in out:
                    checks.append(f'interval branch missing {key}: {out.keys()}')

        # Branch 2: no visibility payload -> plain window fallback
        (evidence / 'visibility_scan.json').write_text(_json.dumps({
            'visibility': {},
            'suggested_bridge_window': {'start_time': 0.0, 'end_time': 2.0},
        }), encoding='utf-8')
        out = tool(context=context, anchor_frame_ids=[])
        if out.get('error'):
            checks.append(f'fallback branch failed: {out["error"]}')
        else:
            for key in ('window_start', 'window_end', 'bridge_frame_ids'):
                if key not in out:
                    checks.append(f'fallback branch missing {key}: {out.keys()}')
    return checks


def _check_bridge_window():
    """The scanned interval must win over a narrowed request.

    Regression: the local model passed the chair's first 8-second interval
    ([0, 8]) instead of the suggested [28, 140].  SAM3 then saw seven frames,
    found one chair and the answer was 1 instead of 2.
    """
    checks = []
    widen = agent.effective_bridge_window
    suggestion = {'start_time': 28.0, 'end_time': 140.0}

    start, end, widened = widen(0.0, 8.0, suggestion)
    if (start, end, widened) != (0.0, 140.0, True):
        checks.append(f'narrow request not widened: {(start, end, widened)}')

    # already covering it: unchanged, nothing to report
    start, end, widened = widen(0.0, 168.0, suggestion)
    if (start, end, widened) != (0.0, 168.0, False):
        checks.append(f'wide request changed: {(start, end, widened)}')

    # explicit opt-out is honoured
    start, end, widened = widen(0.0, 8.0, suggestion, respect_requested=True)
    if (start, end, widened) != (0.0, 8.0, False):
        checks.append(f'opt-out ignored: {(start, end, widened)}')

    # no suggestion recorded yet: leave the request alone
    start, end, widened = widen(0.0, 8.0, None)
    if (start, end, widened) != (0.0, 8.0, False):
        checks.append(f'empty suggestion changed the request: {(start, end)}')

    # a partially populated suggestion must not crash or corrupt the window
    start, end, widened = widen(0.0, 8.0, {'start_time': None})
    if (start, end, widened) != (0.0, 8.0, False):
        checks.append(f'malformed suggestion changed the request: {(start, end)}')

    return checks


def _args(tmp: Path) -> Namespace:
    return Namespace(
        dataset='arkitscenes',
        scene_name='41069025',
        question_id=1,
        data_root=tmp / 'data',
        results_root=tmp / 'results',
        device='cpu',
        max_rounds=5,
        max_steps_per_round=None,
        max_total_steps=15,
        max_turns=None,
        max_tool_failures=2,
        max_api_calls=60,
        allow_tool_generation=False,
        force_plan=False,
        reset_agent_memory=False,
    )


def main():
    failures = _check_bridge_window()
    failures += _check_tools_never_return_none()
    failures += _check_repeat_guard()
    failures += _check_unparseable_planner_output()
    failures += _check_plan_completeness_repair()
    failures += _check_budget_pressure_nudge()
    failures += _check_progress_watchdog()
    failures += _check_sanitize_plan_drops_frame_vocabulary()
    failures += _check_bind_normalises_geometry_less_instances()
    failures += _check_find_bridge_frames_runs()
    if failures:
        print('FAIL (bridge window)')
        for item in failures:
            print('  -', item)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / 'data').mkdir(parents=True, exist_ok=True)

        # A complete planner config so resolve_endpoint() succeeds.  No request
        # is ever made because the loop is stubbed.
        env = {
            'AGENT_PLANNER_MODEL': 'stub-planner',
            'AGENT_PLANNER_BASE_URL': 'https://example.invalid/v1',
            'AGENT_PLANNER_API_KEY': 'sk-stub-not-a-real-key',
        }
        for key in ('AGENT_COT_REASONER_MODEL', 'AGENT_CODE_GENERATOR_MODEL'):
            os.environ.pop(key, None)

        args = _args(tmp)
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(agent, 'parse_args', return_value=args), \
             mock.patch.object(agent, 'load_question', return_value=QUESTION), \
             mock.patch.object(agent, 'QuestionEvidencePlanner', _StubEvidencePlanner), \
             mock.patch.object(agent, 'PlannerLoop', _StubPlannerLoop):
            asyncio.run(agent.main())

        scene_root = tmp / 'results' / 'arkitscenes' / '41069025'
        result_path = scene_root / 'evidence' / 'agent_result.json'
        assert result_path.exists(), f'missing {result_path}'
        result = json.loads(result_path.read_text())
        assert result['done'] is True
        assert result['final_answer'] == '2'
        assert result['llm_endpoints']['planner']['model'] == 'stub-planner'
        assert 'vlm' not in result['llm_endpoints'], 'vlm role should be gone'

        # the plan must have been compiled and saved before the loop ran
        plan_path = scene_root / 'question_plans' / '1.json'
        assert plan_path.exists(), f'missing {plan_path}'
        plan = json.loads(plan_path.read_text())
        assert plan['question_type'] == 'object_counting'
        # The answer key must never be persisted anywhere the agent can read.
        assert 'ground_truth' not in plan, (
            'ground_truth must not be written into the question plan'
        )

        # Re-running a DIFFERENT question in the same scene root must warn:
        # evidence/ and agent_memory.json are per scene root, so questions
        # would share (and corrupt) each other's state.
        evidence_dir = scene_root / 'evidence'
        (evidence_dir / 'agent_result.json').write_text(json.dumps({
            'done': True,
            'task_constraint': {'question_id': 42},
        }), encoding='utf-8')
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(agent, 'parse_args', return_value=args), \
             mock.patch.object(agent, 'load_question', return_value=QUESTION), \
             mock.patch.object(agent, 'QuestionEvidencePlanner', _StubEvidencePlanner), \
             mock.patch.object(agent, 'PlannerLoop', _StubPlannerLoop), \
             contextlib.redirect_stdout(buffer):
            asyncio.run(agent.main())
        output = buffer.getvalue()
        if 'WARNING' not in output or '42' not in output:
            failures.append(
                'a results root reused across questions must warn, got: '
                + output[:200]
            )

        # Even with no agent_result.json, a leftover plan or question dir for
        # another id must be enough to warn.
        (evidence_dir / 'agent_result.json').unlink()
        buffer = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(agent, 'parse_args', return_value=args), \
             mock.patch.object(agent, 'load_question', return_value=QUESTION), \
             mock.patch.object(agent, 'QuestionEvidencePlanner', _StubEvidencePlanner), \
             mock.patch.object(agent, 'PlannerLoop', _StubPlannerLoop), \
             contextlib.redirect_stdout(buffer):
            asyncio.run(agent.main())
        output = buffer.getvalue()
        if 'WARNING' not in output or '42' not in output:
            failures.append(
                'a scene root with another question on disk must warn even '
                'without agent_result.json, got: ' + output[:200]
            )

        constraint_path = scene_root / 'questions' / '1' / 'task_constraint.json'
        assert constraint_path.exists(), f'missing {constraint_path}'
        constraint = json.loads(constraint_path.read_text())
        assert constraint['operation']['operation'] == 'count_instances'
        assert constraint['entities'][0]['category'] == 'chair'

    print('PASS entrypoint smoke (main() wiring, counting question)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
