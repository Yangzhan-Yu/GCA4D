import json
from PIL import Image

def extract_first_integer(s):
    num_str = ''
    for char in s:
        if char.isdigit():
            num_str += char
        elif num_str:
            break
    return int(num_str) if num_str else 0

def run(context, entity, frame_paths):
    max_count = 0
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path)
            data_uri = api.image_to_data_uri(image)
            prompt = f"Count the number of {entity} in the image. Respond with a single integer number only."
            content = [
                {"type": "text", "text": prompt},
                {"type": "image", "data": data_uri}
            ]
            response = api.vlm_chat(context, content, max_tokens=2048)
            try:
                count = int(response.strip())
            except:
                count = extract_first_integer(response)
            if count > max_count:
                max_count = count
        except Exception:
            continue
    return {"count": max_count}

TOOL_SPEC = {
    "name": "count_entities_in_frames",
    "description": "Count unique physical instances of an entity across specified frame paths by querying VLM per frame and taking the maximum observed count.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "frame_paths": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["entity", "frame_paths"]
    }
}
