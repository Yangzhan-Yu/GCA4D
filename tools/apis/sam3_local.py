import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class Sam3TextSegmenter:
    """Thin adapter around the official SAM3 image text-prompt API."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
    ):
        self.checkpoint_path = (
            checkpoint_path
            or os.environ.get('SAM3_CHECKPOINT')
        )
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.model = None
        self.processor = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        if not self.checkpoint_path:
            raise ValueError(
                'SAM3 checkpoint is not configured. Set SAM3_CHECKPOINT or '
                'pass checkpoint_path explicitly.'
            )
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f'SAM3 checkpoint not found: {checkpoint}')

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=self.device,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    @staticmethod
    def _mask_to_numpy(mask) -> np.ndarray:
        if hasattr(mask, 'detach'):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        return mask.astype(bool)

    def segment(
        self,
        image: Image.Image,
        prompts: List[str],
    ) -> Dict[str, List[Dict]]:
        self._ensure_loaded()
        state = self.processor.set_image(image)
        output: Dict[str, List[Dict]] = {prompt: [] for prompt in prompts}

        for prompt in prompts:
            result = self.processor.set_text_prompt(
                prompt=prompt,
                state=state,
            )
            masks = result.get('masks')
            boxes = result.get('boxes')
            scores = result.get('scores')
            if masks is None or boxes is None or scores is None:
                continue
            if hasattr(boxes, 'detach'):
                boxes = boxes.detach().cpu().numpy()
            if hasattr(scores, 'detach'):
                scores = scores.detach().cpu().numpy()
            boxes = np.asarray(boxes)
            scores = np.asarray(scores).reshape(-1)

            for index in range(min(len(scores), len(boxes), len(masks))):
                mask = self._mask_to_numpy(masks[index])
                output[prompt].append({
                    'label': prompt,
                    'bbox': boxes[index].astype(float).tolist(),
                    'score': float(scores[index]),
                    'mask': mask,
                    'mask_area_ratio': float(mask.mean()),
                    'detector': 'sam3',
                })
        return output


__all__ = ['Sam3TextSegmenter']
