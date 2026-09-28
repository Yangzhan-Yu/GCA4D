import tools.apis.generated_tool_api as api
import numpy as np
import cv2
from pathlib import Path

TOOL_SPEC = {
    "name": "generate_room_context",
    "description": "Generate full-frame bounding boxes and SAM masks for the 'room' entity to establish spatial context for counting tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"}
        },
        "required": ["entity"]
    }
}

def run(context, entity):
    if entity != "room":
        return {"error": "This tool only supports 'room' entity"}

    try:
        frame_paths = api.list_memory_frames(context)
        if not frame_paths:
            return {"error": "No frames available in memory"}

        scene_root = Path(api.get_scene_root(context))
        evidence = []

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            bbox = [0, 0, w, h]
            frame_id = Path(frame_path).stem
            
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask_filename = f"room_mask_{frame_id}.png"
            mask_path = scene_root / mask_filename
            cv2.imwrite(str(mask_path), mask)

            evidence.append({
                "frame_id": frame_id,
                "entity": "room",
                "bbox_2d": bbox,
                "mask_path": str(mask_path)
            })

        if not evidence:
            return {"error": "No valid frames processed"}

        api.save_json(context, "room_context_evidence.json", evidence)
        return {
            "status": "success",
            "processed_frames": len(evidence),
            "evidence_file": "room_context_evidence.json"
        }
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}
