"""Bag detection interface.

StubDetector is a no-op placeholder that keeps the pipeline runnable
without a GPU/trained checkpoint. RTMDetDetector wraps the model trained in
../training/ (see its README) — imported lazily inside the class so this
module stays importable in environments without mmdet installed (e.g. this
project's own CPU-only dev machine).

Batched, not mmdet.apis.inference_detector: that convenience function loops
over images one at a time internally even when given a list (each image
gets its own model.test_step() call) — passing it a list is not real GPU
batching. RTMDetDetector builds the batch itself (preprocess each frame
through the model's own test pipeline, then a single model.test_step()
call across the whole batch) so a GPU actually processes multiple frames
in parallel per forward pass, which matters a lot for a 15000-frame video.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import settings


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    label: str = "bag"


class BagDetector(Protocol):
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...


class StubDetector:
    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        return [[] for _ in frames]


class RTMDetDetector:
    def __init__(self, config_path: str, checkpoint_path: str, device: str, score_threshold: float) -> None:
        import torch
        from mmcv.transforms import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        self._torch = torch
        self._model = init_detector(config_path, checkpoint_path, device=device)
        self._score_threshold = score_threshold

        cfg = self._model.cfg.copy()
        test_pipeline_cfg = get_test_pipeline_cfg(cfg)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"  # frames are arrays, not file paths
        self._test_pipeline = Compose(test_pipeline_cfg)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []

        processed = [self._test_pipeline(dict(img=frame, img_id=0)) for frame in frames]
        batch = {
            "inputs": [p["inputs"] for p in processed],
            "data_samples": [p["data_samples"] for p in processed],
        }

        with self._torch.no_grad():
            results = self._model.test_step(batch)

        all_detections = []
        for result in results:
            instances = result.pred_instances
            scores = instances.scores.cpu().numpy()
            bboxes = instances.bboxes.cpu().numpy()
            keep = scores >= self._score_threshold
            all_detections.append(
                [
                    Detection(bbox=tuple(float(v) for v in box), score=float(score))
                    for box, score in zip(bboxes[keep], scores[keep])
                ]
            )
        return all_detections


_detector: BagDetector | None = None


def get_detector() -> BagDetector:
    """Lazily builds and caches the process-wide detector.

    Loading an RTMDet checkpoint is expensive (GPU memory + model init), so
    this is done once and reused across jobs rather than per-video.
    """
    global _detector
    if _detector is None:
        if settings.detector_config_path and settings.detector_checkpoint_path:
            _detector = RTMDetDetector(
                config_path=str(settings.detector_config_path),
                checkpoint_path=str(settings.detector_checkpoint_path),
                device=settings.detector_device,
                score_threshold=settings.detection_score_threshold,
            )
        else:
            _detector = StubDetector()
    return _detector
