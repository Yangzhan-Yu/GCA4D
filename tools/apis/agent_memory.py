from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AgentMemoryEvent:
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    def __init__(self):
        self.events: List[AgentMemoryEvent] = []

    def add(self, event_type: str, **data):
        self.events.append(AgentMemoryEvent(event_type=event_type, data=data))

    def to_dict(self) -> Dict[str, Any]:
        return {'events': [asdict(event) for event in self.events]}

    def prompt_summary(
        self,
        max_events: int = 12,
        max_chars_per_event: int = 1200,
        max_total_chars: int = 8000,
    ) -> Dict[str, Any]:
        """A bounded view of the memory for the Planner prompt.

        The full memory grows with every step (a single visibility scan can be
        9k characters), and it is re-sent on every call.  Left unbounded the
        prompt passes the model's context limit part-way through a question.
        Only the most recent events are kept, long payloads are truncated, and
        the whole thing is capped.

        ``planner_raw`` events are dropped entirely: their content is already
        captured by the matching ``planner_decision``.
        """
        events = [e for e in self.events if e.event_type != 'planner_raw']
        selected = events[-max_events:] if max_events > 0 else events
        dropped = len(events) - len(selected)

        summary: List[Dict[str, Any]] = []
        total = 0
        for event in selected:
            payload = json.dumps(event.data, ensure_ascii=False, default=str)
            truncated = False
            if len(payload) > max_chars_per_event:
                payload = payload[:max_chars_per_event] + '...<truncated>'
                truncated = True
            entry = {
                'event_type': event.event_type,
                'data': payload if truncated else event.data,
            }
            if truncated:
                entry['truncated'] = True
            size = len(payload)
            if total + size > max_total_chars and summary:
                dropped += 1
                continue
            total += size
            summary.append(entry)

        return {
            'events': summary,
            'omitted_events': dropped,
            'note': (
                'Truncated view. The full log is in agent_memory.json on disk; '
                'do not re-derive state from here, read the tool results you '
                'have already seen.'
            ),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    @classmethod
    def load(cls, path: str | Path) -> 'AgentMemory':
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        memory = cls()
        for event in data.get('events', []):
            memory.events.append(AgentMemoryEvent(
                event_type=event['event_type'],
                data=event.get('data', {}),
            ))
        return memory


__all__ = ['AgentMemoryEvent', 'AgentMemory']
