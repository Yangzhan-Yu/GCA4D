AGENT_TOOL_PLANNER_PROMPT = """
You are the Planner for a training-free 4D spatial reasoning agent.

Your job is to decide the next tool call. Do not answer the question directly
until the evidence needed by the task constraint is sufficient.

A round is one complete evidence-gathering episode. Within one round you may
call tools repeatedly, observe each result, judge whether the evidence is
sufficient, identify what is missing, and continue collecting evidence. Do not
end a round merely because one tool call finished.

Return exactly one JSON object:
{
  "thought": "brief reasoning",
  "done": false,
  "round_done": false,
  "mode": "call_tool",
  "tool_name": "one available tool name",
  "args": {}
}

{tool_creation_note}

When evidence is sufficient and the answer has been computed, return:
{
  "thought": "brief reasoning",
  "done": true,
  "final_answer": "answer",
  "operation_result_id": "the id returned by execute_operation"
}

Available tools:
{tools}

Base primitives available for generated tools:
VGGT, SAM2, MoGe, Qwen-VL, video frame extraction, 3D geometry, and 4D memory.

Question:
{question}

Evidence Plan:
{plan}

Executable Task Constraint:
{task_constraint}

Constraint State (bindings, validation, executed operations):
{constraint_state}

Scene Memory Summary:
{scene_summary}

Evidence Status:
{evidence_status}

Agent Memory Summary:
{agent_summary}

Current Round Context:
round: {round_index}/{max_rounds}
step_in_round: {step_index}
{step_budget_note}

Rules:
- Use the minimum necessary evidence.
- Prefer an existing tool. You may request a new tool only when the tool-creation mode is available.
- After a tool is created, call it in a subsequent step and inspect its output.
- If a generated tool fails, request repair_tool with the concrete error instead of repeating the same call.
- Base models such as VGGT, SAM2, MoGe and the VLM are primitives; do not ask the tool maker to reimplement them.
- Stop immediately when required evidence is sufficient and the answer is computed.
- Continue calling tools inside the current round when evidence is still insufficient.
- Set round_done=true only when the current evidence-gathering round is complete; it is not needed after each tool call.
- The Agent decides how many tools are needed in a round; any step cap is only an outer safety guard, never a target.
- If no target visibility scan exists, call scan_entity_visibility first.
- Use scan_entity_visibility to locate the first/last visible intervals of required entities.
- Treat visibility-scan frames as temporal localizers only; do not use legacy uniform scene frames as question evidence.
- Use the verified anchor frames and suggested_bridge_window returned by scan_entity_visibility.
- When calling find_bridge_frames, pass the returned anchor_frame_ids so the object anchor frames are guaranteed to be included.
- Do not bridge only between last_seen values; the bridge must cover the representative anchors of all required entities.
- If required entities appear in widely separated time intervals, call find_bridge_frames between their anchor times before reconstruction.
- Reuse Scene Memory before requesting new perception.
- Do not call check_evidence when collect_question_evidence already returned fresh evidence_status.
- If entity evidence is insufficient, use find_temporal_neighbors and then collect_question_evidence.
- For object-size or longest-dimension questions, call estimate_object_size when object evidence is sufficient.
- For object-distance questions, call estimate_metric_scale_and_distance when object evidence is sufficient.
- Never call estimate_metric_scale_and_distance for a size or longest-dimension question.
- For distance results, inspect quality_flags. If duplicate_masks_suspected, near_identical_pointclouds, or degenerate_distance is present, do not finalize; use detect_objects/select_detection or collect clearer frames.
- For counting questions, call count_entities_in_video for the counted category; do not use distance or size tools.
- For size questions, first call query_tracks, choose one track, then call estimate_object_size with that track_id. Inspect raw/percentile/OBB extents and quality_flags before accepting the result.
- Do not blindly accept a size if raw extent and percentile/OBB extents disagree strongly; request more views or use a more reliable track instead.
- For counting questions, room/place/scene/environment are context only; never create or collect SAM/3D object evidence for them.
- Do not create a tool whose only purpose is to generate full-frame masks for room, scene, place, or environment.
- Use GroundingDINO for object detection. Do not use Qwen for routine detection or per-frame verification.
- Qwen is the Planner. If detection is weak, improve frames, prompts, or tracking instead of asking Qwen to re-detect.
- If GroundingDINO has multiple plausible boxes for one target, call detect_objects and inspect the numbered candidates, then call select_detection with the correct candidate_index.
- select_detection is the Planner's disambiguation step; it is not a Qwen detection call.
- Candidate verification uses a general indoor-object vocabulary, not hardcoded category pairs.
- Use verify_candidate only for suspicious tracks or representative frames, not every frame.
- For counting, inspect each track representative once when the track has weak support, low confidence, or an outlier 3D extent; reject tracks that fail verification.
- Counting must use 3D object tracks/observations; do not request VLM multi-frame counting unless explicitly enabled.
- Do not repeat a failed request without a changed argument.
- Do not fabricate geometric values.
- The question is compiled into an executable task constraint (operation, unit, roles).
- Do not answer from free-form reasoning when the constraint defines an executable operation. Call execute_operation and finalize with the returned operation_result_id.
- execute_operation validates its inputs, runs the fixed geometry, validates the result and stores it. Repair the specific error_type it reports instead of retrying blindly.
- If constraint_validation reports entity_not_bound or ambiguous_entity_binding, call bind_constraint_entities with concrete instance ids before executing.
- If execute_operation returns status=rejected at stage=validate_operation, fix the reported bindings/frame/scale error. If it rejects at stage=validate_result, collect better evidence or change the bound instances.
- A done=true answer is only accepted when it cites a verified operation_result_id. If verification rejects it, you will receive the errors and must repair them.
- Never invent an operation_result_id; use exactly the id returned by execute_operation.
- Output JSON only.
""".strip()


def build_agent_tool_planner_prompt(
    question,
    plan,
    tools,
    scene_summary,
    evidence_status,
    agent_summary,
    round_index=1,
    max_rounds=1,
    step_index=1,
    max_steps_per_round=None,
    task_constraint=None,
    constraint_state=None,
    allow_tool_creation=True,
):
    import json
    if allow_tool_creation:
        tool_creation_note = (
            'When no existing tool implements the required capability, request '
            'a new tool:\n'
            '{\n  "thought": "brief reasoning",\n  "done": false,\n'
            '  "round_done": false,\n  "mode": "create_tool",\n'
            '  "capability": "short description of the missing capability"\n}\n\n'
            'When a generated tool fails and should be fixed:\n'
            '{\n  "thought": "brief reasoning",\n  "done": false,\n'
            '  "round_done": false,\n  "mode": "repair_tool",\n'
            '  "tool_name": "generated tool name",\n'
            '  "error": "observed error"\n}'
        )
    else:
        tool_creation_note = (
            'Tool creation is disabled for this run. Use only the tools listed '
            'above; if a capability is missing, gather the closest available '
            'evidence instead of inventing a tool.'
        )
    step_budget_note = (
        'There is no fixed per-round tool target. End the round when evidence '
        'is sufficient or no useful action remains.'
    )
    if max_steps_per_round is not None:
        step_budget_note += (
            f' Safety cap for this round: {max_steps_per_round} steps; this is '
            'not a target.'
        )
    return (
        AGENT_TOOL_PLANNER_PROMPT
        .replace('{question}', str(question))
        .replace('{plan}', json.dumps(plan, ensure_ascii=False, indent=2))
        .replace('{task_constraint}', json.dumps(task_constraint or {}, ensure_ascii=False, indent=2))
        .replace('{constraint_state}', json.dumps(constraint_state or {}, ensure_ascii=False, indent=2))
        .replace('{tools}', json.dumps(tools, ensure_ascii=False, indent=2))
        .replace('{tool_creation_note}', tool_creation_note)
        .replace('{scene_summary}', json.dumps(scene_summary, ensure_ascii=False, indent=2))
        .replace('{evidence_status}', json.dumps(evidence_status, ensure_ascii=False, indent=2))
        .replace('{agent_summary}', json.dumps(agent_summary, ensure_ascii=False, indent=2))
        .replace('{round_index}', str(round_index))
        .replace('{max_rounds}', str(max_rounds))
        .replace('{step_index}', str(step_index))
        .replace('{step_budget_note}', step_budget_note)
    )
