EVIDENCE_PLANNER_PROMPT = """
You are a spatial evidence planner. Analyze the current question and decide
which evidence must be acquired. Do not enumerate unrelated scene objects.

Return exactly one JSON object with this schema:
{{
  "task_type": "counting | metric_measurement | relative_direction | relative_distance | route_planning | appearance_order | other",
  "target_entities": ["entity needed for the answer"],
  "reference_entities": ["entity that defines the reference frame or comparison"],
  "relation_or_metric": "the relation or quantity to compute",
  "time_constraint": null or a time/order description,
  "reference_frame": "world | camera | object:<name> | direction:<name> | unknown",
  "required_evidence": [
    "2d_grounding",
    "sam_mask",
    "object_track",
    "camera_pose",
    "3d_points",
    "metric_scale",
    "scene_graph",
    "event_timeline"
  ],
  "reasoning": "brief reason for the requested evidence"
}}

Rules:
- Only include entities that are needed for this question.
- For a counting question, target_entities contains the counted category.
- For a counting question, do not put room, place, scene, or environment in reference_entities. Those are context, not objects.
- For a counting question, set reference_frame to world unless a concrete object defines the reference.
- For a distance or size question, request metric_scale and 3d_points.
- For a direction question, request the relevant reference entities and 3d_points.
- For an appearance-order question, request event_timeline and object_track.
- Do not request all categories in the scene.
- Do not request SAM masks, 3D points, or object observations for room, place, scene, or environment.
- Output JSON only.

Question type: {question_type}
Question: {question}
Options: {options}
""".strip()


def build_evidence_planner_prompt(question, question_type=None, options=None):
    if options:
        options_text = '\n'.join(str(option) for option in options)
    else:
        options_text = 'None'
    return EVIDENCE_PLANNER_PROMPT.format(
        question=question,
        question_type=question_type or 'unknown',
        options=options_text,
    )
