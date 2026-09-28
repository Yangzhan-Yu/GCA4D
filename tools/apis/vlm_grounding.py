import base64
import io
import json
import re
from typing import Dict, List

from PIL import Image, ImageDraw

from tools.apis.api_budget import reserve_api_call


_CATEGORY_ALIASES = {
    'sofa': {'sofa', 'couch', 'settee'},
    'stove': {'stove', 'oven', 'range', 'cooker', 'cooktop', 'hob'},
    'refrigerator': {'refrigerator', 'fridge', 'freezer'},
    'television': {'television', 'tv', 'monitor'},
    'chair': {'chair', 'seat'},
    'table': {'table', 'desk'},
    'bed': {'bed', 'mattress'},
}


def category_matches(requested: str, actual: str) -> bool:
    requested = requested.strip().lower()
    actual = actual.strip().lower()
    aliases = _CATEGORY_ALIASES.get(requested, {requested})
    return actual in aliases or any(alias in actual for alias in aliases)


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def parse_multi_entity_detections(content: str, image: Image.Image) -> Dict[str, Dict]:
    match = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL)
    if not match:
        return {}
    try:
        records = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}

    width, height = image.size
    candidates: Dict[str, List] = {}
    for record in records:
        label = str(record.get('label', '')).strip().lower()
        bbox = record.get('bbox_2d')
        if not label or not bbox or len(bbox) != 4:
            continue
        x1 = int(float(bbox[0]) / 1000 * width)
        y1 = int(float(bbox[1]) / 1000 * height)
        x2 = int(float(bbox[2]) / 1000 * width)
        y2 = int(float(bbox[3]) / 1000 * height)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidates.setdefault(label, []).append({
            'bbox': [x1, y1, x2, y2],
            'area': area,
            'confidence': float(record.get('confidence', 1.0)),
            'reason': str(record.get('reason', '')),
        })

    detections = {}
    for label, items in candidates.items():
        best = max(items, key=lambda item: item['area'])
        detections[label] = {
            'bbox': best['bbox'],
            'score': best['confidence'],
            'reason': best['reason'],
        }
    return detections


def parse_verification(content: str):
    match = re.search(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL)
    raw = match.group(1) if match else content
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return False, 0.0, 'invalid_json'
    return (
        bool(record.get('valid', False)),
        float(record.get('confidence', 0.0)),
        str(record.get('category', 'unknown')),
    )


def verify_detection_vlm(
    client,
    model: str,
    image: Image.Image,
    entity: str,
    bbox,
):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = [int(value) for value in bbox]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=5)
    prompt = (
        f'The red box is proposed as a "{entity}". Is it actually a fully '
        f'visible, unambiguous {entity} with enough surface visible for 3D '
        'measurement? Reject furniture fragments, floors, walls, pictures, '
        'reflections, or regions that merely resemble the target. Return one '
        'JSON object in a json code block: '
        '{"valid": true/false, "category": "...", "confidence": 0.0-1.0, '
        '"reason": "..."}'
    )
    reserve_api_call('vlm_verify_detection', {'entity': entity})
    response = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': image_to_data_uri(annotated)}},
            ],
        }],
        max_tokens=512,
        temperature=0.0,
        top_p=0.95,
    )
    return parse_verification(response.choices[0].message.content)


def detect_entities_vlm(
    client,
    model: str,
    image: Image.Image,
    entities: List[str],
    verify_detections: bool = False,
    verification_attempts: int = 1,
):
    category_text = ', '.join(f'"{entity}"' for entity in entities)
    prompt = (
        f'Locate every visible instance for the categories: {category_text}. '
        'Return a JSON array. Each item must have this format: '
        '{"bbox_2d": [x1, y1, x2, y2], "label": "category", '
        '"confidence": 0.0-1.0, "reason": "brief reason"}. '
        'Treat common synonyms as the requested category, for example '
        'sofa/couch and stove/oven/range/cooker/cooktop. Use the requested '
        'category name in the label. Coordinates must be normalized to '
        '0-1000. If a category is not visible, omit it.'
    )
    reserve_api_call('vlm_detect_entities', {'entities': entities})
    response = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': image_to_data_uri(image)}},
            ],
        }],
        max_tokens=1024,
        temperature=0.0,
        top_p=0.95,
    )
    detections = parse_multi_entity_detections(
        response.choices[0].message.content, image
    )
    if not verify_detections:
        return detections

    verified = {}
    allowed = {str(entity).strip().lower() for entity in entities}
    for entity, detection in detections.items():
        if entity not in allowed:
            continue
        for _ in range(max(1, verification_attempts)):
            valid, confidence, category = verify_detection_vlm(
                client,
                model,
                image,
                entity,
                detection['bbox'],
            )
            if valid and category_matches(entity, category):
                verified[entity] = {
                    **detection,
                    'score': confidence,
                    'verified_category': entity,
                }
                break
    return verified


__all__ = [
    'image_to_data_uri',
    'category_matches',
    'parse_multi_entity_detections',
    'detect_entities_vlm',
    'verify_detection_vlm',
]
