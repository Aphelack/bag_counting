"""Cross-frame tracking and counting.

Placeholder: the real tracking method (e.g. IoU/ByteTrack + a counting-line
crossing rule) and duplicate-count protection are still to be decided. This
class is the extension point the processing pipeline calls into.
"""
from app.detector import Detection


class BagCounter:
    def __init__(self) -> None:
        self.total = 0

    def update(self, detections: list[Detection], frame_idx: int) -> int:
        return self.total
