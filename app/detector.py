"""Bag detection interface.

The production detector wraps an MMDetection model (choice of architecture
and checkpoint is still TBD). StubDetector is a no-op placeholder that keeps
the rest of the pipeline runnable end-to-end in the meantime.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


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
