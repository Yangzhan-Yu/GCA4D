import numpy as np
import csv
from pathlib import Path
import json

TOOL_SPEC = {
    "name": "align_arkitscenes_trajectory",
    "description": "Align reconstructed camera trajectory to ARKitScenes metric trajectory using Procrustes analysis to compute metric scale factor in meters per reconstruction unit",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

def run(context, **args):
    try:
        scene_root = api.get_scene_root(context)
        sensor_data_path = Path(scene_root) / "sensor_data.csv"
        recon_path = Path(scene_root) / "generated_frames" / "camera_poses.json"

        # Load ARKitScenes ground truth positions
        gt_positions = []
        with open(sensor_data_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            pos_x_idx = headers.index('pos_x')
            pos_y_idx = headers.index('pos_y')
            pos_z_idx = headers.index('pos_z')
            for row in reader:
                gt_positions.append([
                    float(row[pos_x_idx]),
                    float(row[pos_y_idx]),
                    float(row[pos_z_idx])
                ])
        gt_positions = np.array(gt_positions)

        # Load reconstructed camera positions
        with open(recon_path, 'r') as f:
            recon_data = json.load(f)
        recon_positions = np.array([
            [pose['position'][0], pose['position'][1], pose['position'][2]]
            for pose in recon_data['frames']
        ])

        # Center both trajectories
        gt_centered = gt_positions - gt_positions.mean(axis=0)
        recon_centered = recon_positions - recon_positions.mean(axis=0)

        # Compute scale factor via Procrustes (ratio of Frobenius norms)
        gt_norm = np.linalg.norm(gt_centered)
        recon_norm = np.linalg.norm(recon_centered)
        scale_factor = gt_norm / recon_norm

        result = {
            "scale_factor": float(scale_factor),
            "units": "meters per reconstruction unit",
            "ground_truth_frame_count": len(gt_positions),
            "reconstruction_frame_count": len(recon_positions)
        }
        api.save_json(context, "procrustes_scale.json", result)
        return result

    except Exception as e:
        return {
            "error": f"Trajectory alignment failed: {str(e)}",
            "scale_factor": None
        }
