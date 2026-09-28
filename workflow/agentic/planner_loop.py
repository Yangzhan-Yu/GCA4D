import json
from typing import Any, Callable, Dict, Optional

from workflow.agentic.tool_executor import ToolExecutor
from workflow.agentic.tool_registry import ToolRegistry
from tools.apis.api_budget import ApiBudgetExceeded, reserve_api_call
from workflow.prompts.agent_tool_planner import build_agent_tool_planner_prompt
from workflow.utils.parse_utils import parse_first_json_object


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
        return self.agent_memory.to_dict()

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
                try:
                    reserve_api_call(
                        'planner',
                        {
                            'round': round_index + 1,
                            'step': step_index,
                        },
                    )
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=2048,
                        temperature=0.0,
                        top_p=0.95,
                    )
                except ApiBudgetExceeded as exc:
                    return {
                        'done': False,
                        'error': str(exc),
                        'rounds': round_index + 1,
                        'steps': total_steps,
                    }
                content = response.choices[0].message.content
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
                except Exception:
                    print('[Planner] Failed to parse decision. Raw response follows:', flush=True)
                    print(content, flush=True)
                    raise

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
        }
