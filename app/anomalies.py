"""Anomaly monitoring for the counting pipeline.

Placeholder: what qualifies as an anomaly (e.g. detector confidence
collapse, long zero-detection gaps, implausible count jumps) is still to be
decided. This class is the extension point the processing pipeline calls
into.
"""
from app.detector import Detection
from app.models import Anomaly


class AnomalyMonitor:
    def check(self, detections: list[Detection], frame_idx: int, fps: float) -> list[Anomaly]:
        return []
