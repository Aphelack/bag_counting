"""Cross-frame tracking and belt counting.

Greedy IoU tracker with simple constant-velocity prediction, plus a
counting-zone crossing rule: a rectangular zone instead of a single line,
since a zone tolerates a frame or two of missed detection right at the
boundary, where a thin line would just miss the count entirely. Each track
is counted at most once — the first time its centroid enters the zone —
which is what prevents double-counting the same bag as it drifts across
several frames.

Prediction matters even for slow, smooth belt motion: without it, a track
that goes a few frames without a matching detection (a brief occlusion, a
confidence dip) sits frozen at its last known position while the real bag
keeps moving. By the time detection resumes, the bag has drifted out of
IoU range of the frozen box, so it looks like a brand new object — spawning
a second track that gets counted again on top of the first. Advancing each
unmatched track's box by its last known velocity every frame keeps it near
where the bag actually is, so matching survives short gaps instead of
silently double-counting.

The default zone is tuned to this project's fixed camera angle
(storage/input/input.mp4: a diagonal conveyor belt running from the
upper-middle of the frame to the lower-left) — pass a different
`zone_fractional` for another camera setup.
"""
from dataclasses import dataclass

from app.detector import Detection

Bbox = tuple[float, float, float, float]  # x1, y1, x2, y2
Vector = tuple[float, float]  # dx, dy per frame

# Fractional (x1, y1, x2, y2) of frame size, covering a clean cross-section
# of the belt: past the dark tunnel opening (avoids partial-occlusion
# misses right as bags emerge) and before the accumulation pile at the
# bottom-left (avoids double-counting overlapping bags there).
DEFAULT_ZONE_FRACTIONAL: Bbox = (0.05, 0.40, 0.80, 0.62)


def iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def centroid(bbox: Bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def shift(bbox: Bbox, velocity: Vector) -> Bbox:
    dx, dy = velocity
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def in_zone(point: tuple[float, float], zone: Bbox) -> bool:
    x, y = point
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= x <= zx2 and zy1 <= y <= zy2


@dataclass
class Track:
    id: int
    bbox: Bbox
    velocity: Vector = (0.0, 0.0)
    age_since_seen: int = 0
    counted: bool = False


class BagCounter:
    def __init__(
        self,
        zone_fractional: Bbox = DEFAULT_ZONE_FRACTIONAL,
        iou_threshold: float = 0.2,
        max_age: int = 10,
    ) -> None:
        self.total = 0
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._zone_fractional = zone_fractional
        self._zone: Bbox | None = None

    def set_frame_size(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self._zone_fractional
        self._zone = (x1 * width, y1 * height, x2 * width, y2 * height)

    @property
    def zone(self) -> Bbox | None:
        return self._zone

    def update(self, detections: list[Detection], frame_idx: int) -> int:
        if self._zone is None:
            raise RuntimeError("call set_frame_size() before update()")

        # Advance every track's box by its last known velocity before
        # matching, so a track that's gone unseen for a few frames is
        # compared against where the bag should be now, not where it was
        # last actually observed.
        predicted = {tid: shift(t.bbox, t.velocity) for tid, t in self._tracks.items()}

        # Greedy IoU matching against the predicted boxes: score every
        # (track, detection) pair, keep those above threshold, then assign
        # highest-scoring pairs first.
        candidates = []
        for track_id, pred_bbox in predicted.items():
            for det_idx, det in enumerate(detections):
                score = iou(pred_bbox, det.bbox)
                if score >= self._iou_threshold:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _score, track_id, det_idx in candidates:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(det_idx)
            track = self._tracks[track_id]
            old_cx, old_cy = centroid(track.bbox)
            new_bbox = detections[det_idx].bbox
            new_cx, new_cy = centroid(new_bbox)
            # Velocity from the last *observed* position, not the
            # predicted one — avoids compounding prediction error while a
            # track is intermittently missed over several frames.
            track.velocity = (new_cx - old_cx, new_cy - old_cy)
            track.bbox = new_bbox
            track.age_since_seen = 0

        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.bbox = predicted[track_id]
                track.age_since_seen += 1
        self._tracks = {tid: t for tid, t in self._tracks.items() if t.age_since_seen <= self._max_age}

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                self._tracks[self._next_id] = Track(id=self._next_id, bbox=det.bbox)
                self._next_id += 1

        for track in self._tracks.values():
            if not track.counted and in_zone(centroid(track.bbox), self._zone):
                track.counted = True
                self.total += 1

        return self.total
