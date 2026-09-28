import json
import re
from pathlib import Path
from typing import Any, Dict, List

from workflow.agentic.tool_registry import ToolSpec
from workflow.agentic.tool_validator import validate_generated_tool
from workflow.utils.parse_utils import parse_json_str
from tools.apis.api_budget import reserve_api_call


SYNTHESIZER_PROMPT = """
You are the Tool Maker for a training-free 4D spatial reasoning agent.

Create one Python tool that implements the missing capability. The tool will be
validated and then executed in an isolated subprocess. Return JSON only:
{
  "thought": "brief design rationale",
  "code": "complete Python source code"
}

The generated module must define:
1. TOOL_SPEC: a JSON-serializable dict with keys name, description, parameters.
   parameters must be a JSON object, for example:
   {"type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"]}
2. def run(context, **args): returns a JSON-serializable dict.

You may only import these modules:
- json, math, statistics, collections, itertools, pathlib, typing
- numpy, cv2, PIL, matplotlib
- tools.apis.generated_tool_api as api

Useful base API functions:
{base_api}

Do not import subprocess, socket, requests, urllib, shutil, or arbitrary filesystem
helpers. Do not call eval, exec, compile, __import__, or input. Do not run shell
commands. Use api.save_json(context, filename, value) for outputs. For VLM image
messages, use api.image_content(path_or_image) and never construct raw
{"type": "image", ...} dictionaries. Do not use bare except; catch concrete
exceptions or return the error details.

Existing tools:
{existing_tools}

Question:
{question}

Evidence Plan:
{plan}

Missing capability:
{capability}

Repair feedback:
{feedback}
""".strip()


def _slugify(value: str) -> str:
    value = re.sub(r'[^a-zA-Z0-9_]+', '_', value.strip().lower())
    value = re.sub(r'_+', '_', value).strip('_')
    if not value:
        value = 'generated_tool'
    if value[0].isdigit():
        value = f'tool_{value}'
    return value


def _extract_code(content: str) -> str:
    match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', content, re.DOTALL)
    if match:
        return match.group(1).strip() + '\n'
    return content.strip() + '\n'


def load_generated_tool_specs(generated_dir: str | Path):
    generated_dir = Path(generated_dir)
    manifest_path = generated_dir / 'registry.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    specs = []
    for name, record in manifest.get('tools', {}).items():
        source_path = record.get('source_path')
        if not source_path or not Path(source_path).exists():
            continue
        validation = validate_generated_tool(source_path)
        if not validation.get('valid'):
            continue
        tool_spec = validation['tool_spec']
        specs.append(ToolSpec(
            name=tool_spec.get('name', name),
            description=tool_spec.get('description', record.get('description', name)),
            parameters=tool_spec.get('parameters', record.get('parameters', {})),
            source_path=str(source_path),
            metadata={
                **record.get('metadata', {}),
                'generated': True,
                'loaded_from_manifest': str(manifest_path),
            },
        ))
    return specs


class ToolSynthesizer:
    def __init__(
        self,
        client,
        model: str,
        generated_dir: str | Path,
    ):
        self.client = client
        self.model = model
        self.generated_dir = Path(generated_dir)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.generated_dir / 'registry.json'

    def _persist_spec(self, spec: ToolSpec):
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        else:
            manifest = {'tools': {}}
        manifest.setdefault('tools', {})[spec.name] = {
            'name': spec.name,
            'description': spec.description,
            'parameters': spec.parameters,
            'source_path': spec.source_path,
            'metadata': spec.metadata,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    async def synthesize(
        self,
        capability: str,
        question: str,
        plan: Dict[str, Any],
        existing_tools: List[Dict[str, Any]],
        feedback: str = '',
    ) -> ToolSpec:
        base_api = (
            'api.open_memory(context), api.list_memory_frames(context), '
            'api.list_memory_frame_paths(context), '
            'api.get_question_evidence(context, question_id), '
            'api.sample_video_frames(context, max_frames=12, subdir="generated_frames"), '
            'api.image_content(image_or_path), api.image_to_data_uri(image), '
            'api.vlm_chat(context, content, max_tokens=2048), '
            'api.parse_json_object(content), api.save_json(context, filename, value), '
            'api.run_entrypoint(context, module, args) for allowed VGGT/SAM2/MoGe '
            'entrypoints, '
            'api.get_scene_root(context)'
        )
        prompt = (
            SYNTHESIZER_PROMPT
            .replace('{base_api}', base_api)
            .replace('{existing_tools}', json.dumps(existing_tools, ensure_ascii=False, indent=2))
            .replace('{question}', question)
            .replace('{plan}', json.dumps(plan, ensure_ascii=False, indent=2))
            .replace('{capability}', capability)
            .replace('{feedback}', feedback or 'None')
        )
        reserve_api_call('tool_synthesizer', {'capability': capability})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=4096,
            temperature=0.0,
            top_p=0.95,
        )
        content = response.choices[0].message.content
        decision, _ = parse_json_str(content)
        code = _extract_code(str(decision.get('code', '')))
        slug = _slugify(capability)
        path = self.generated_dir / f'{slug}.py'
        path.write_text(code, encoding='utf-8')

        validation = validate_generated_tool(path)
        if not validation.get('valid'):
            errors = '\n'.join(validation.get('errors', []))
            raise RuntimeError(f'Generated tool validation failed: {errors}')
        tool_spec = validation['tool_spec']
        spec = ToolSpec(
            name=tool_spec['name'],
            description=tool_spec.get('description', capability),
            parameters=tool_spec.get('parameters', {
                'type': 'object',
                'properties': {},
            }),
            source_path=str(path),
            metadata={
                'generated': True,
                'capability': capability,
                'synthesizer_model': self.model,
                'validation': validation,
            },
        )
        self._persist_spec(spec)
        return spec

    async def repair(
        self,
        tool_spec: ToolSpec,
        question: str,
        plan: Dict[str, Any],
        existing_tools: List[Dict[str, Any]],
        error: str,
    ) -> ToolSpec:
        source = ''
        if tool_spec.source_path and Path(tool_spec.source_path).exists():
            source = Path(tool_spec.source_path).read_text(encoding='utf-8')
        repair_context = (
            f'Repair existing tool {tool_spec.name}. Previous error:\n{error}\n'
            f'Previous source code:\n```python\n{source}\n```'
        )
        return await self.synthesize(
            capability=tool_spec.metadata.get('capability', tool_spec.name),
            question=question,
            plan=plan,
            existing_tools=existing_tools,
            feedback=repair_context,
        )


__all__ = ['ToolSynthesizer', 'load_generated_tool_specs']
