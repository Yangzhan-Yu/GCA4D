import numpy as np
from pathlib import Path
import api

tool_spec = {
    "name": "estimate_metric_scale_from_trajectory",
    "description": "Estimate metric scale factor by comparing ground truth camera trajectory with reconstructed trajectory",
    "parameters": {
        "type": "object",
        "properties": {
            "resolution_level": {
                "type": "integer",
                "default": 7
            }
        },
        "required": []
    }
}

def run(context, resolution_level=7):
    try:
        scene_root = api.get_scene_root(context)
        scene_path = Path(scene_root)

        # Load ground truth ARKit camera poses (in meters)
        gt_poses_path = scene_path / "gt_camera_poses.npy"
        if not gt_poses_path.exists():
            return {"error": f"Ground truth poses not found at {gt_poses_path}"

        # Load reconstructed camera poses from VGGT
        recon_poses_path = scene_path / "reconstruction" / "camera_poses.npy"
        if not recon_poses_path.exists():
            return {"error": f"Reconstructed poses not found at {recon_poses_path}"

        gt_poses = np.load(gt_poses_path)
        recon_poses = np.load(recon_poses_path)

        # Calculate real-world distance traveled
        real_total = 0.0
        for i in range(1, len(gt_poses)):
            pos1 = gt_poses[i-1][:3, 3]
            pos2 = gt_poses[i][:3, 3]
            real_total += np.linalg.norm(pos2 - pos1)

        # Calculate reconstructed trajectory distance
        recon_total = 0.0
        for i in range(1, len(recon_poses)):
            pos1 = recon_poses[i-1][:3, 3]
            pos2 = recon_poses[i][:3, 3]
            recon_total += np.linalg.norm(pos2 - pos1)

        if recon_total <= 1e-6:
            return {"error": "Reconstructed trajectory has zero length"

        scale = real_total / recon_total
        return {"scale": float(scale)}

    except Exception as e:
        return {"error": str(e)}
