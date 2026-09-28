from typing import Dict, List

import torch
import torchvision
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


GROUNDING_DINO_ID = 'IDEA-Research/grounding-dino-base'
CACHE_DIR = '/data3/Agentic-Spatial-Reasoning/hf_cache/hub'


class GroundingDinoDetector:
    def __init__(
        self,
        device: str = 'cuda',
        threshold: float = 0.2,
        text_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ):
        self.device = device
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        ).to(device).eval()

    @torch.inference_mode()
    def detect_all(self, image: Image.Image, entities: List[str]) -> Dict[str, List[Dict]]:
        detections = {entity: [] for entity in entities}
        for entity in entities:
            inputs = self.processor(
                images=image,
                text=[entity],
                return_tensors='pt',
            ).to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            boxes = result['boxes'].detach().cpu()
            scores = result['scores'].detach().cpu()
            if boxes.numel() == 0:
                continue
            keep = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            boxes, scores = boxes[keep], scores[keep]
            if boxes.numel() == 0:
                continue
            order = torch.argsort(scores, descending=True)
            for index in order:
                detections[entity].append({
                    'bbox': boxes[index].tolist(),
                    'score': float(scores[index]),
                    'label': entity,
                    'detector': 'grounding_dino',
                })
        return detections

    def detect(self, image: Image.Image, entities: List[str]) -> Dict[str, Dict]:
        all_detections = self.detect_all(image, entities)
        detections = {}
        for entity, candidates in all_detections.items():
            if not candidates:
                continue
            best = max(candidates, key=lambda item: item['score'])
            detections[entity] = {
                **best,
                'verified_category': entity,
            }
        return detections


__all__ = ['GroundingDinoDetector']
