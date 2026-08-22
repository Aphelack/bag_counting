"""Run a trained RTMDet checkpoint on a video and draw detections.

This validates the trained detector in isolation — per-frame bag detections
drawn on an output video, plus summary stats. It does NOT do cross-frame
tracking or belt counting: that algorithm is still an open decision (see
`app/tracker.py`), so this script only reports what the detector sees in
each frame, not a deduplicated conveyor count.

Not runnable in this dev environment (no GPU, no mmdet installed); meant to
run on the GPU host after training.

Usage:
    uv run python scripts/infer.py \\
        configs/rtmdet_bag.py work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth \\
        --video ../storage/input/input.mp4 --out ../storage/output/rtmdet_preview.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to MMDetection config used for training")
    parser.add_argument("checkpoint", help="path to trained .pth checkpoint")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="annotated output video path")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def draw_detections(frame_bgr: np.ndarray, bboxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    vis = frame_bgr.copy()
    for (x1, y1, x2, y2), score in zip(bboxes.astype(int), scores):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(vis, f"detections: {len(scores)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def main() -> None:
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    counts_per_frame: list[int] = []
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            result = inference_detector(model, frame_bgr)
            instances = result.pred_instances
            keep = instances.scores.cpu().numpy() >= args.score_thr
            bboxes = instances.bboxes.cpu().numpy()[keep]
            scores = instances.scores.cpu().numpy()[keep]

            writer.write(draw_detections(frame_bgr, bboxes, scores))
            counts_per_frame.append(len(scores))

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"frame {frame_idx} | detections: {len(scores)}")
    finally:
        cap.release()
        writer.release()

    counts = np.array(counts_per_frame)
    zero_frames = int((counts == 0).sum())
    print(f"\nprocessed {len(counts)} frames -> {args.out}")
    print(f"detections/frame: mean={counts.mean():.2f} min={counts.min()} max={counts.max()}")
    print(f"frames with zero detections: {zero_frames} ({100 * zero_frames / len(counts):.1f}%)")


if __name__ == "__main__":
    main()
