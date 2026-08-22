"""Bootstrap-label a COCO dataset for bag detection using SAM 3.

Samples frames from a conveyor video and runs SAM 3 text-prompt segmentation
on each one (native `facebookresearch/sam3` API — same calls as
`../notebooks/01_sam3_segmentation_test.ipynb`, just batched over the whole
video instead of a handful of manually inspected frames). Bounding boxes are
derived from the returned masks (not `output["boxes"]`) because the mask
coordinate space is unambiguous, while the native repo's box convention
wasn't confirmed against this SAM 3 build.

Written to run on a CUDA host with the sam3 package + checkpoint already
available, matching the environment used in the notebook:

    uv pip install git+https://github.com/facebookresearch/sam3.git

Not runnable in this dev environment (no GPU, no sam3 checkpoint) — intended
to be run on the remote GPU host, hence the CLI plumbing instead of hardcoded
paths.

Output layout (COCO format, ready for `training/configs/rtmdet_bag.py`):

    <out-dir>/
      images/frame{idx:06d}.jpg
      annotations/train.json
      annotations/val.json
      previews/frame{idx:06d}.jpg   (subset, boxes drawn, for manual QA)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_PROMPT = "A white soft pillow positioned on an industrial roller conveyor system."
CATEGORY_NAME = "bag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="path to input.mp4")
    parser.add_argument("--out-dir", type=Path, required=True, help="output dataset directory")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="SAM3 checkpoint (sam3.pt)")
    parser.add_argument("--bpe-path", type=Path, required=True, help="bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="SAM3 text prompt")
    parser.add_argument("--stride", type=int, default=25, help="label every Nth frame (25 @ 25fps = 1/sec)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="SAM3Processor threshold")
    parser.add_argument("--min-mask-area", type=int, default=200, help="drop masks smaller than this, px^2")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="fraction of frames held out for val")
    parser.add_argument("--preview-every", type=int, default=20, help="save 1 in N labeled frames to previews/")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def iter_sampled_frames(video_path: Path, stride: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break
        yield frame_idx, frame_bgr
        frame_idx += stride

    cap.release()


def load_processor(checkpoint_path: Path, bpe_path: Path, device: str, confidence_threshold: float):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible; run this on the GPU host")

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
    )
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def masks_to_boxes(masks: torch.Tensor, min_area: int) -> list[tuple[int, int, int, int]]:
    """masks: [N, 1, H, W] -> list of (x, y, w, h) tight boxes, tiny masks dropped."""
    boxes = []
    for i in range(masks.shape[0]):
        mask = masks[i, 0].detach().float().cpu().numpy() > 0.5
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w, h = x2 - x1 + 1, y2 - y1 + 1
        if w * h < min_area:
            continue
        boxes.append((x1, y1, w, h))
    return boxes


def label_frame(processor, frame_bgr: np.ndarray, prompt: str, min_area: int) -> list[tuple[int, int, int, int]]:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)
    return masks_to_boxes(output["masks"], min_area)


def draw_preview(frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    vis = frame_bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(vis, f"{len(boxes)} boxes", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def main() -> None:
    args = parse_args()

    images_dir = args.out_dir / "images"
    annotations_dir = args.out_dir / "annotations"
    previews_dir = args.out_dir / "previews"
    for d in (images_dir, annotations_dir, previews_dir):
        d.mkdir(parents=True, exist_ok=True)

    processor = load_processor(args.checkpoint_path, args.bpe_path, args.device, args.confidence_threshold)

    labeled = []  # list of dicts: {frame_idx, width, height, boxes}
    for i, (frame_idx, frame_bgr) in enumerate(iter_sampled_frames(args.video, args.stride)):
        boxes = label_frame(processor, frame_bgr, args.prompt, args.min_mask_area)
        h, w = frame_bgr.shape[:2]
        file_name = f"frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(images_dir / file_name), frame_bgr)
        labeled.append({"frame_idx": frame_idx, "file_name": file_name, "width": w, "height": h, "boxes": boxes})

        if i % args.preview_every == 0:
            cv2.imwrite(str(previews_dir / file_name), draw_preview(frame_bgr, boxes))

        print(f"frame {frame_idx:6d} -> {len(boxes)} boxes")

    if not labeled:
        raise SystemExit("no frames labeled — check --video and --stride")

    split_idx = int(len(labeled) * (1 - args.val_ratio))
    write_coco(labeled[:split_idx], annotations_dir / "train.json")
    write_coco(labeled[split_idx:], annotations_dir / "val.json")

    n_boxes = sum(len(entry["boxes"]) for entry in labeled)
    print(
        f"\nlabeled {len(labeled)} frames ({n_boxes} boxes total, "
        f"{n_boxes / len(labeled):.2f} avg/frame) -> {args.out_dir}"
    )
    print(f"train: {split_idx} frames | val: {len(labeled) - split_idx} frames")
    print(f"previews saved for manual review -> {previews_dir}")


def write_coco(entries: list[dict], out_path: Path) -> None:
    images, annotations = [], []
    ann_id = 1
    for img_id, entry in enumerate(entries, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": entry["file_name"],
                "width": entry["width"],
                "height": entry["height"],
            }
        )
        for x, y, w, h in entry["boxes"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CATEGORY_NAME}],
    }
    out_path.write_text(json.dumps(coco))
    print(f"wrote {len(images)} images / {len(annotations)} annotations -> {out_path}")


if __name__ == "__main__":
    main()
