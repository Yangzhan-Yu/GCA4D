from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional

from workflow.prompts.evidence_planner import build_evidence_planner_prompt
from workflow.utils.parse_utils import parse_json_str
from tools.apis.api_budget import reserve_api_call


@dataclass
class EvidenceRequest:
    task_type: str
    target_entities: List[str] = field(default_factory=list)
    reference_entities: List[str] = field(default_factory=list)
    relation_or_metric: str = ''
    time_constraint: Optional[Any] = None
    reference_frame: str = 'unknown'
    required_evidence: List[str] = field(default_factory=list)
    reasoning: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_evidence_request(content: str) -> EvidenceRequest:
    if not isinstance(content, str):
        content = str(content)
    parsed, _ = parse_json_str(content)

    required_keys = [
        'task_type',
        'target_entities',
        'reference_entities',
        'relation_or_metric',
        'reference_frame',
        'required_evidence',
    ]
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise ValueError(f'Evidence plan missing keys: {missing}')

    if not isinstance(parsed['target_entities'], list):
        raise TypeError('target_entities must be a list')
    if not isinstance(parsed['reference_entities'], list):
        raise TypeError('reference_entities must be a list')
    if not isinstance(parsed['required_evidence'], list):
        raise TypeError('required_evidence must be a list')

    target_entities = [str(item) for item in parsed['target_entities']]
    reference_entities = [str(item) for item in parsed['reference_entities']]
    required_evidence = {str(item) for item in parsed['required_evidence']}

    # Expand implicit dependencies so downstream tools always have prerequisites.
    if target_entities or reference_entities:
        if '3d_points' in required_evidence or 'metric_scale' in required_evidence:
            required_evidence.update({'2d_grounding', 'sam_mask', '3d_points'})
        if 'object_track' in required_evidence:
            required_evidence.add('2d_grounding')
    if 'metric_scale' in required_evidence:
        required_evidence.add('3d_points')

    return EvidenceRequest(
        task_type=str(parsed['task_type']),
        target_entities=target_entities,
        reference_entities=reference_entities,
        relation_or_metric=str(parsed['relation_or_metric']),
        time_constraint=parsed.get('time_constraint'),
        reference_frame=str(parsed['reference_frame']),
        required_evidence=sorted(required_evidence),
        reasoning=str(parsed.get('reasoning', '')),
    )


class QuestionEvidencePlanner:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    async def plan(
        self,
        question: str,
        question_type: Optional[str] = None,
        options: Optional[List[str]] = None,
    ) -> EvidenceRequest:
        prompt = build_evidence_planner_prompt(
            question=question,
            question_type=question_type,
            options=options,
        )
        reserve_api_call('evidence_planner')
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=2048,
            temperature=0.0,
            top_p=0.95,
        )
        content = response.choices[0].message.content
        return parse_evidence_request(content)


__all__ = [
    'EvidenceRequest',
    'QuestionEvidencePlanner',
    'parse_evidence_request',
]
