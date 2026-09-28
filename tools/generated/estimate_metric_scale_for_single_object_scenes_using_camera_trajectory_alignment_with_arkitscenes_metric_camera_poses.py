import json
import math
import statistics
from pathlib import Path

tool_spec = {
    "name": "align_camera_trajectory_for_scale",
    "description": "Estimate metric scale by aligning reconstructed camera trajectory with ARKitScenes metric camera poses. Saves the scale factor for use in object size estimation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

def run(context, **args):
    try:
        scene_root = api.get_scene_root(context)
        if not scene_root:
            return {"error": "Scene root not found"}

        scene_path = Path(scene_root)
        arkit_poses_path = scene_path / "poses.txt"
        recon_poses_path = scene_path / "reconstructed_poses.json"

        # Load ARKit metric poses
        arkit_poses = []
        if not arkit_poses_path.exists():
            return {"error": "ARKit poses file not found"}
        with open(arkit_poses_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        tx, ty, tz = map(float, parts[1:4])
                        arkit_poses.append((tx, ty, tz))
                    except (ValueError, IndexError):
                        continue
        if len(arkit_poses) < 2:
            return {"error": "Insufficient ARKit pose data"}

        # Load reconstructed poses
        if not recon_poses_path.exists():
            return {"error": "Reconstructed poses file not found"}
        with open(recon_poses_path, 'r') as f:
            recon_data = json.load(f)
        recon_poses = []
        if isinstance(recon_data, list):
            recon_poses = recon_data
        elif isinstance(recon_data, dict) and 'positions' in recon_data:
            recon_poses = recon_data['positions']
        else:
            recon_poses = [frame.get('position', []) for frame in recon_data]
        recon_poses = [pos for pos in recon_poses if len(pos) == 3]
        if len(recon_poses) < 2:
            return {"error": "Insufficient reconstructed pose data"}

        # Compute pairwise distances
        def get_distances(positions):
            return [
                math.sqrt(sum((a - b)**2 for a, b in zip(positions[i], positions[i-1])))
                for i in range(1, len(positions))
            ]

        arkit_dists = get_distances(arkit_poses)
        recon_dists = get_distances(recon_poses)

        # Calculate scale factor
        n = min(len(arkit_dists), len(recon_dists))
        if n < 1:
            return {"error": "No valid segments for scale computation"}

        ratios = []
        for i in range(n):
            if recon_dists[i] > 0.001:
                ratios.append(arkit_dists[i] / recon_dists[i])
        
        if not ratios:
            return {"error": "No valid scale ratios computed"}

        scale_factor = statistics.median(ratios)
        result = {"scale_factor": scale_factor}
        api.save_json(context, "metric_scale.json", result)
        return {"status": "success", "scale_factor": scale_factor}

    except Exception as e:
        return {"error": f"Scale estimation failed: {str(e)}"}
