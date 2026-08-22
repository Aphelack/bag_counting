"""Video processing pipeline: detect, track/count, annotate, write output.

Runs synchronously on a worker thread (see main.py) so it never blocks the
event loop.
"""
import logging
from pathlib import Path

import cv2

from app.anomalies import AnomalyMonitor
from app.detector import BagDetector, get_detector
from app.jobs import job_store
from app.models import JobStatus
from app.tracker import BagCounter

logger = logging.getLogger(__name__)

PROGRESS_UPDATE_EVERY_N_FRAMES = 30


def run(job_id: str, input_path: Path, output_path: Path, detector: BagDetector | None = None) -> None:
    detector = detector or get_detector()
    counter = BagCounter()
    monitor = AnomalyMonitor()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    counter.set_frame_size(width, height)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    anomalies = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            counter.update(detections, frame_idx)
            anomalies.extend(monitor.check(detections, frame_idx, fps))
            _draw_overlay(frame, detections, counter.total, counter.zone)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % PROGRESS_UPDATE_EVERY_N_FRAMES == 0:
                job_store.update(
                    job_id,
                    progress=min(frame_idx / total_frames, 1.0),
                    bag_count=counter.total,
                )
    finally:
        cap.release()
        writer.release()

    job_store.update(
        job_id,
        status=JobStatus.COMPLETED,
        progress=1.0,
        bag_count=counter.total,
        anomalies=anomalies,
    )


def _draw_overlay(frame, detections, bag_count: int, zone: tuple[float, float, float, float] | None) -> None:
    if zone is not None:
        zx1, zy1, zx2, zy2 = (int(v) for v in zone)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 1)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    cv2.putText(
        frame,
        f"Bags: {bag_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
