import json
from collections import Counter
from typing import Any, Callable, Dict, Optional

from tools.apis.llm_endpoint import (
    async_chat_text,
    planner_max_tokens,
    request_extra,
)
from workflow.agentic.tool_executor import ToolExecutor
from workflow.agentic.tool_registry import ToolRegistry
from tools.apis.api_budget import ApiBudgetExceeded, reserve_api_call
from workflow.prompts.agent_tool_planner import build_agent_tool_planner_prompt
from workflow.utils.parse_utils import parse_first_json_object


def operation_name_from_state(state) -> str:
    """The configured operation, from either shape of ``task_constraint``.

    The loop's ``state_provider`` summarises the constraint as
    ``{'operation': 'count_instances', ...}`` (a string), while the full
    compiled constraint is ``{'operation': {'operation': 'count_instances'}}``
    (a dict).  Code that assumed one shape broke on the other - accepting both
    here is cheaper than keeping two shapes in sync across modules.
    """
    constraint = (state or {}).get('task_constraint') or {}
    if not isinstance(constraint, dict):
        return ''
    operation = constraint.get('operation')
    if isinstance(operation, dict):
        operation = operation.get('operation')
    return str(operation) if operation else ''


class PlannerLoop:
    def __init__(
        self,
        client,
        model: str,
        registry: ToolRegistry,
        context: Dict[str, Any],
        agent_memory=None,
        max_rounds: int = 5,
        max_steps_per_round: Optional[int] = None,
        max_total_steps: Optional[int] = 25,
        max_turns: Optional[int] = None,
        max_tool_failures: int = 2,
        state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        tool_synthesizer=None,
        done_validator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.client = client
        self.model = model
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.context = context
        self.agent_memory = agent_memory
        self.max_rounds = max(1, int(max_rounds))
        if max_turns is not None:
            max_steps_per_round = max_turns
        self.max_steps_per_round = (
            None
            if max_steps_per_round is None
            else max(1, int(max_steps_per_round))
        )
        self.max_total_steps = (
            None if max_total_steps is None else max(1, int(max_total_steps))
        )
        self.max_tool_failures = max_tool_failures
        self.tool_failure_counts = {}
        # Identical (tool, args) repeats are the signature of a loop.  The step
        # cap eventually stops the run, but by then every step is wasted, so
        # the third identical call is refused with an explicit instruction.
        self.call_signatures = Counter()
        # A local planner occasionally emits prose or an unclosed  thinking
        # block instead of the decision JSON.  That is an expected failure mode
        # and must not abort the whole question: without a result file the
        # official evaluator has no prediction to score at all.
        self.max_parse_failures = 2
        self.parse_failures = 0
        # Distinguishes "we cut it off" from "it was stuck".  A run that ends
        # while still making progress means the budget was the binding
        # constraint, which is exactly what must not happen on free local
        # inference; a run that ends flat is a reasoning problem.
        self.max_steps_without_progress = max(8, self.max_total_steps // 3 if self.max_total_steps else 8)
        self._best_progress = -1
        self._steps_since_progress = 0
        self._last_progress_note = ''
        self.state_provider = state_provider
        self.tool_synthesizer = tool_synthesizer
        self.done_validator = done_validator

    def _state(self) -> Dict[str, Any]:
        if self.state_provider is None:
            return {}
        return self.state_provider()

    def _memory_summary(self):
        if self.agent_memory is None:
            return {}
        return self.agent_memory.prompt_summary()

    def _available_tools(self):
        tools = self.registry.describe()
        question_type = self.context.get('question_type')
        if question_type == 'object_size_estimation':
            tools = [
                tool for tool in tools
                if tool['name'] != 'estimate_metric_scale_and_distance'
            ]
        elif question_type == 'object_abs_distance':
            tools = [
                tool for tool in tools
                if tool['name'] != 'estimate_object_size'
            ]
        return tools

    @staticmethod
    def _progress_score(state) -> int:
        """A coarse measure of how close the constraint is to being answerable.

        Weights are ordered by how decisive each event is: a verified result
        ends the question, an operation result is one validation away, a valid
        constraint means every role is bound.
        """
        validation = state.get('constraint_validation') or {}
        resolved = len(validation.get('resolved_bindings') or {})
        return (
            100 * len(state.get('verified_operation_results') or {})
            + 20 * len(state.get('operation_results') or {})
            + 10 * int(bool(validation.get('valid')))
            + resolved
        )

    def _track_progress(self, state) -> bool:
        """Update the progress tracker; return True if this step improved it."""
        score = self._progress_score(state)
        if score > self._best_progress:
            self._best_progress = score
            self._steps_since_progress = 0
            return True
        self._steps_since_progress += 1
        return False

    def _budget_pressure_note_safe(self, state, total_steps: int) -> str:
        """Wrap the note so a bug here cannot abort a question.

        This runs on the prompt path: an exception used to propagate out of the
        loop and kill the process with no result file at all.
        """
        try:
            return self._budget_pressure_note(state, total_steps)
        except Exception as exc:  # noqa: BLE001
            print(
                f'[Planner] budget note skipped ({type(exc).__name__}: {exc})',
                flush=True,
            )
            return ''

    def _budget_pressure_note(self, state, total_steps: int) -> str:
        """Push a wandering Planner to the executable step before the cap.

        Observed repeatedly: the evidence was already sufficient, but the
        Planner kept re-checking detections and never called execute_operation,
        so the run ended with no answer at all.  The nudge is driven by the
        step budget rather than by tuning the standing prompt text.
        """
        if not self.max_total_steps:
            return ''
        operation = operation_name_from_state(state)
        if not operation:
            return ''
        used = total_steps / max(1, self.max_total_steps)
        if used < 0.6:
            return ''
        verified = state.get('verified_operation_results') or {}
        executed = state.get('operation_results') or {}
        if verified:
            return (
                f'\n\n*** BUDGET: {total_steps}/{self.max_total_steps} steps used. '
                f'A verified result already exists ({sorted(verified)}). '
                'Finalize NOW with done=true and its operation_result_id. ***'
            )
        if executed:
            return (
                f'\n\n*** BUDGET: {total_steps}/{self.max_total_steps} steps used. '
                f'Operation results exist ({sorted(executed)}) but none is '
                'verified. Repair the reported error_type, or collect the '
                'missing evidence, then finalize. ***'
            )
        return (
            f'\n\n*** BUDGET: {total_steps}/{self.max_total_steps} steps used and '
            f'no operation result exists yet. The constraint defines '
            f'{operation!r}. Call execute_operation now with the evidence you '
            'already have (bind any missing roles first), then finalize with '
            'its operation_result_id. Do not keep re-checking detections. ***'
        )

    async def run(self, question: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        total_steps = 0
        for round_index in range(self.max_rounds):
            step_index = 0
            while True:
                if (
                    self.max_total_steps is not None
                    and total_steps >= self.max_total_steps
                ):
                    return {
                        'done': False,
                        'error': (
                            'Planner exceeded max total steps: '
                            f'{self.max_total_steps}'
                        ),
                        'rounds': round_index + 1,
                        'steps': total_steps,
                        # If this is 0 the run was still improving when it was
                        # cut off - i.e. the budget was the binding constraint.
                        'steps_since_progress': self._steps_since_progress,
                        'progress_score': self._best_progress,
                    }
                if (
                    self.max_steps_per_round is not None
                    and step_index >= self.max_steps_per_round
                ):
                    print(
                        f'[Planner] Round {round_index + 1} step safety cap '
                        f'({self.max_steps_per_round}) reached; starting next round.',
                        flush=True,
                    )
                    break
                step_index += 1
                total_steps += 1
                print(
                    f'\n[Planner] Round {round_index + 1}/{self.max_rounds} '
                    f'Step {step_index}',
                    flush=True,
                )
                state = self._state()
                self._track_progress(state)
                if self._steps_since_progress >= self.max_steps_without_progress:
                    return {
                        'done': False,
                        'error': (
                            'No measurable progress for '
                            f'{self._steps_since_progress} steps (no new bound '
                            'role, operation result or verified result). The '
                            'Planner is stuck, not out of budget.'
                        ),
                        'rounds': round_index + 1,
                        'steps': total_steps,
                        'progress_score': self._best_progress,
                        'steps_since_progress': self._steps_since_progress,
                    }
                print(
                    '[Planner] Scene state: '
                    f"frames={state.get('scene_summary', {}).get('counts', {}).get('frames', '?')}, "
                    f"objects={state.get('scene_summary', {}).get('counts', {}).get('objects', '?')}",
                    flush=True,
                )
                prompt = build_agent_tool_planner_prompt(
                    question=question,
                    plan=plan,
                    tools=self._available_tools(),
                    scene_summary=state.get('scene_summary', {}),
                    evidence_status=state.get('evidence_status', {}),
                    agent_summary=self._memory_summary(),
                    round_index=round_index + 1,
                    max_rounds=self.max_rounds,
                    step_index=step_index,
                    max_steps_per_round=self.max_steps_per_round,
                    allow_tool_creation=self.tool_synthesizer is not None,
                    task_constraint=state.get('task_constraint', {}),
                    constraint_state={
                        'bindings': state.get('constraint_bindings', {}),
                        'validation': state.get('constraint_validation', {}),
                        'operation_results': state.get('operation_results', {}),
                        'verified_operation_results': state.get(
                            'verified_operation_results', {}
                        ),
                    },
                )
                prompt += self._budget_pressure_note_safe(state, total_steps)
                try:
                    reserve_api_call(
                        'planner',
                        {
                            'round': round_index + 1,
                            'step': step_index,
                        },
                    )
                    content = await async_chat_text(
                        self.client,
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=planner_max_tokens(2048),
                        temperature=0.0,
                        top_p=0.95,
                        extra=request_extra(),
                    )
                except ApiBudgetExceeded as exc:
                    return {
                        'done': False,
                        'error': str(exc),
                        'rounds': round_index + 1,
                        'steps': total_steps,
                    }
                except Exception as exc:  # noqa: BLE001 - provider/model errors
                    print(
                        f'[Planner] API call failed: {type(exc).__name__}: {exc}',
                        flush=True,
                    )
                    return {
                        'done': False,
                        'error': f'Planner API call failed: {type(exc).__name__}: {exc}',
                        'rounds': round_index + 1,
                        'steps': total_steps,
                    }
                print(f'[Planner] Raw response:\n{content}', flush=True)
                if self.agent_memory is not None:
                    self.agent_memory.add(
                        'planner_raw',
                        round=round_index,
                        step=step_index,
                        content=content,
                    )
                try:
                    decision = parse_first_json_object(content)
                    self.parse_failures = 0
                except Exception as exc:  # noqa: BLE001
                    self.parse_failures += 1
                    print(
                        '[Planner] Unparseable decision '
                        f'({self.parse_failures}/{self.max_parse_failures}); '
                        f'raw response starts: {content[:200]!r}',
                        flush=True,
                    )
                    if self.agent_memory is not None:
                        self.agent_memory.add(
                            'planner_parse_failure',
                            round=round_index,
                            step=step_index,
                            content=content[:2000],
                            error=str(exc)[:500],
                        )
                    if self.parse_failures >= self.max_parse_failures:
                        return {
                            'done': False,
                            'error': (
                                'Planner returned output that is not a valid '
                                f'decision JSON {self.parse_failures} times; '
                                f'last response started with {content[:120]!r}'
                            ),
                            'rounds': round_index + 1,
                            'steps': total_steps,
                            'last_raw_response': content[:2000],
                        }
                    # Retry once with an explicit correction.  A plain retry
                    # would reproduce the same output at temperature 0, so the
                    # message has to change.
                    try:
                        reserve_api_call('planner_corrective_retry')
                        content = await async_chat_text(
                            self.client,
                            model=self.model,
                            messages=[
                                {'role': 'user', 'content': prompt},
                                {
                                    'role': 'user',
                                    'content': (
                                        'Your previous reply was not a valid '
                                        'decision JSON object.  Reply with ONLY '
                                        'the JSON object: no prose, no '
                                        'markdown fence, no thinking block.'
                                    ),
                                },
                            ],
                            max_tokens=planner_max_tokens(2048),
                            temperature=0.0,
                            top_p=0.95,
                            extra=request_extra(),
                        )
                        decision = parse_first_json_object(content)
                        self.parse_failures = 0
                    except ApiBudgetExceeded as budget_exc:
                        return {
                            'done': False,
                            'error': str(budget_exc),
                            'rounds': round_index + 1,
                            'steps': total_steps,
                        }
                    except Exception as retry_exc:  # noqa: BLE001
                        return {
                            'done': False,
                            'error': (
                                'Planner output was not a valid decision JSON '
                                'and the corrective retry failed: '
                                f'{type(retry_exc).__name__}: {retry_exc}'
                            ),
                            'rounds': round_index + 1,
                            'steps': total_steps,
                            'last_raw_response': content[:2000],
                        }

                if decision.get('thought'):
                    print(f"[Planner] Thought: {decision['thought']}", flush=True)

                if self.agent_memory is not None:
                    self.agent_memory.add(
                        'planner_decision',
                        round=round_index,
                        step=step_index,
                        decision=decision,
                    )

                if decision.get('done'):
                    final_answer = decision.get('final_answer')
                    verification = None
                    if self.done_validator is not None:
                        try:
                            verification = self.done_validator(decision)
                        except Exception as exc:  # noqa: BLE001
                            verification = {
                                'accepted': False,
                                'reason': f'done_validator raised {type(exc).__name__}: {exc}',
                            }
                        if not verification.get('accepted'):
                            print(
                                '[Planner] Final answer rejected by constraint '
                                f"verification: {verification.get('reason')}",
                                flush=True,
                            )
                            if self.agent_memory is not None:
                                self.agent_memory.add(
                                    'done_rejected',
                                    round=round_index,
                                    step=step_index,
                                    decision=decision,
                                    verification=verification,
                                )
                            continue
                        final_answer = verification.get('final_answer', final_answer)
                    print(
                        f'[Planner] Done. final_answer={final_answer}',
                        flush=True,
                    )
                    return {
                        'done': True,
                        'final_answer': final_answer,
                        'rounds': round_index + 1,
                        'steps': total_steps,
                        'decision': decision,
                        'verification': verification,
                    }

                if decision.get('round_done'):
                    print(
                        f'[Planner] Round {round_index + 1} complete; '
                        'starting a new round if needed.',
                        flush=True,
                    )
                    break

                mode = decision.get('mode', 'call_tool')
                if mode == 'create_tool':
                    capability = (
                        decision.get('capability')
                        or decision.get('missing_capability')
                        or decision.get('thought')
                    )
                    print(
                        f'[ToolMaker] Creating capability: {capability}',
                        flush=True,
                    )
                    if self.tool_synthesizer is None:
                        result = {'error': 'Tool synthesis is not configured'}
                    else:
                        try:
                            spec = await self.tool_synthesizer.synthesize(
                                capability=str(capability),
                                question=question,
                                plan=plan,
                                existing_tools=self.registry.describe(),
                            )
                            self.registry.register(spec, replace=True)
                            result = {
                                'created_tool': spec.name,
                                'description': spec.description,
                                'parameters': spec.parameters,
                                'source_path': spec.source_path,
                            }
                            print(
                                f'[ToolMaker] Registered {spec.name}: {spec.source_path}',
                                flush=True,
                            )
                        except Exception as exc:
                            result = {
                                'error': f'Tool synthesis failed: {exc}',
                                'capability': capability,
                            }
                    if self.agent_memory is not None:
                        self.agent_memory.add(
                            'tool_creation',
                            round=round_index,
                            step=step_index,
                            capability=capability,
                            result=result,
                        )
                    print(
                        f'[ToolMaker] Result keys={list(result.keys())}',
                        flush=True,
                    )
                    continue

                if mode == 'repair_tool':
                    tool_name = decision.get('tool_name')
                    print(
                        f'[ToolMaker] Repairing tool: {tool_name}',
                        flush=True,
                    )
                    if self.tool_synthesizer is None:
                        result = {'error': 'Tool synthesis is not configured'}
                    elif not tool_name:
                        result = {'error': 'repair_tool requires tool_name'}
                    else:
                        try:
                            existing = self.registry.get(tool_name)
                            spec = await self.tool_synthesizer.repair(
                                tool_spec=existing,
                                question=question,
                                plan=plan,
                                existing_tools=self.registry.describe(),
                                error=str(decision.get('error', 'unknown error')),
                            )
                            self.registry.register(spec, replace=True)
                            result = {
                                'repaired_tool': spec.name,
                                'source_path': spec.source_path,
                            }
                        except Exception as exc:
                            result = {
                                'error': f'Tool repair failed: {exc}',
                                'tool_name': tool_name,
                            }
                    if self.agent_memory is not None:
                        self.agent_memory.add(
                            'tool_repair',
                            round=round_index,
                            step=step_index,
                            tool_name=tool_name,
                            result=result,
                        )
                    print(
                        f'[ToolMaker] Result keys={list(result.keys())}',
                        flush=True,
                    )
                    continue

                tool_name = decision.get('tool_name')
                args = decision.get('args', {})
                signature = (
                    str(tool_name),
                    json.dumps(args or {}, sort_keys=True, default=str),
                )
                repeat_count = self.call_signatures[signature]
                if repeat_count >= 2:
                    print(
                        f'[Tool] Refusing repeat #{repeat_count + 1} of '
                        f'{tool_name} with identical arguments.',
                        flush=True,
                    )
                    result = {
                        'error': 'repeated_call_refused',
                        'tool_name': tool_name,
                        'times_already_called': repeat_count,
                        'message': (
                            'This exact call has already been made '
                            f'{repeat_count} times with the same arguments. '
                            'Change the arguments (different frames, entity or '
                            'window) or use a different tool.'
                        ),
                    }
                    if self.agent_memory is not None:
                        self.agent_memory.add(
                            'tool_observation',
                            round=round_index,
                            step=step_index,
                            tool_name=tool_name,
                            args=args,
                            result=result,
                        )
                    continue
                self.call_signatures[signature] += 1
                print(
                    f'[Tool] Executing {tool_name} with args={args}',
                    flush=True,
                )
                result = self.executor.execute(
                    tool_name=tool_name,
                    args=args,
                    context=self.context,
                )
                failed = bool(result.get('error')) or (
                    'returncode' in result and result.get('returncode') != 0
                )
                if failed:
                    if (
                        result.get('budget_exceeded')
                        or 'API budget exceeded' in str(result.get('error', ''))
                    ):
                        return {
                            'done': False,
                            'error': str(result.get('error', 'API budget exceeded')),
                            'rounds': round_index + 1,
                            'steps': total_steps,
                            'last_tool_result': result,
                        }
                    self.tool_failure_counts[tool_name] = (
                        self.tool_failure_counts.get(tool_name, 0) + 1
                    )
                    print(
                        f'[Tool] Failed {tool_name}; consecutive failures='
                        f"{self.tool_failure_counts[tool_name]}",
                        flush=True,
                    )
                    if self.tool_failure_counts[tool_name] >= self.max_tool_failures:
                        return {
                            'done': False,
                            'error': (
                                f'Tool {tool_name} failed '
                                f'{self.tool_failure_counts[tool_name]} times; stopping.'
                            ),
                            'rounds': round_index + 1,
                            'steps': total_steps,
                            'last_tool_result': result,
                        }
                else:
                    self.tool_failure_counts[tool_name] = 0

                print(
                    f'[Tool] Completed {tool_name}. Result keys={list(result.keys())}',
                    flush=True,
                )
                if self.agent_memory is not None:
                    self.agent_memory.add(
                        'tool_observation',
                        round=round_index,
                        step=step_index,
                        tool_name=tool_name,
                        args=args,
                        result=result,
                    )

        return {
            'done': False,
            'error': (
                f'Planner exceeded max rounds: {self.max_rounds}'
            ),
            'rounds': self.max_rounds,
            'steps': total_steps,
            'steps_since_progress': self._steps_since_progress,
            'progress_score': self._best_progress,
        }
