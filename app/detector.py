"""Bag detection interface.

StubDetector is a no-op placeholder that keeps the pipeline runnable
without a GPU/trained checkpoint. RTMDetDetector wraps the model trained in
../training/ (see its README) via mmdet's inference API — imported lazily
inside the class so this module stays importable in environments without
mmdet installed (e.g. this project's own CPU-only dev machine).
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
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class StubDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []


class RTMDetDetector:
    def __init__(self, config_path: str, checkpoint_path: str, device: str, score_threshold: float) -> None:
        from mmdet.apis import init_detector

        self._model = init_detector(config_path, checkpoint_path, device=device)
        self._score_threshold = score_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        from mmdet.apis import inference_detector

        result = inference_detector(self._model, frame)
        instances = result.pred_instances
        scores = instances.scores.cpu().numpy()
        bboxes = instances.bboxes.cpu().numpy()
        keep = scores >= self._score_threshold
        return [
            Detection(bbox=tuple(float(v) for v in box), score=float(score))
            for box, score in zip(bboxes[keep], scores[keep])
        ]


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
