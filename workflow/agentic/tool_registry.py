from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Dict[str, Any]] = None
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, replace: bool = False):
        if not spec.name:
            raise ValueError('Tool name must not be empty')
        if spec.handler is None and spec.source_path is None:
            raise ValueError(f'Tool {spec.name} has no handler')
        if spec.name in self._tools and not replace:
            raise ValueError(f'Tool already registered: {spec.name}')
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f'Unknown tool: {name}')
        return self._tools[name]

    def list_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': spec.name,
                'description': spec.description,
                'parameters': spec.parameters,
                'source': 'generated' if spec.source_path else 'builtin',
            }
            for spec in self.list_specs()
        ]
