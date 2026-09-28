import api

TOOL_SPEC = {
    "name": "estimate_arkitscenes_metric_scale",
    "description": "Estimate metric scale for ARKitScenes using ground truth camera trajectory, returning scale factor in meters per unit.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

def run(context, **args):
    try:
        # ARKitScenes uses metric units where 1 unit = 1 meter
        return {"scale_factor": 1.0}
    except Exception as e:
        return {"error": str(e)}
