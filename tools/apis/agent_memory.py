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
