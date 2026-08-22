# Training: RTMDet bag detector

Builds the MMDetection-based bag detector: bootstrap-label a dataset from
`input.mp4` with SAM 3, fine-tune RTMDet-tiny on it, then validate the
checkpoint. Everything here runs on a CUDA host — nothing in this directory
runs on the dev machine this was written on (no GPU there).

## 1. Setup

```bash
cd training
uv sync
# mmcv needs a build matched to your installed torch/CUDA — install via mim,
# not through uv/pyproject, so it can pick the right prebuilt wheel:
uv run mim install mmengine "mmcv>=2.0.0,<2.2.0" mmdet
```

Labeling (step 2) also needs the `sam3` package and its assets, in whichever
environment you're already running `../experiments/notebooks/01_sam3_segmentation_test.ipynb`
in:

```bash
uv pip install git+https://github.com/facebookresearch/sam3.git
# checkpoint_path / bpe_path from the notebook — same files, reused here.
```

## 2. Label a dataset from input.mp4

```bash
uv run --project ../experiments python ../experiments/scripts/label_with_sam3.py \
    --video ../storage/input/input.mp4 \
    --out-dir data/bags_coco \
    --checkpoint-path /path/to/sam3.pt \
    --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --stride 25 \
    --confidence-threshold 0.3
```

This samples one frame per second (`--stride 25` @ 25fps), runs SAM 3 with
the prompt `"A white soft pillow positioned on an industrial roller conveyor
system."` (the one that worked in testing — override with `--prompt` to try
others), derives bounding boxes from the returned masks, and writes a COCO
dataset split chronologically into train/val (last 15% of frames held out,
to avoid near-duplicate frames leaking between splits).

**Check `data/bags_coco/previews/` before training** — every 20th labeled
frame is saved there with boxes drawn, so you can eyeball recall/false
positives (e.g. the "clothes on the floor" confusion from earlier testing)
before spending GPU time training on bad labels. If quality is poor for a
subset of frames, the cheapest fix is usually deleting those images and
their annotations from the COCO JSON rather than re-running the whole
pipeline.

## 3. Train

```bash
uv run python scripts/train.py configs/rtmdet_bag.py
```

Fine-tunes from COCO-pretrained RTMDet-tiny weights (`load_from` in the
config) for 50 epochs, single class (`bag`). Checkpoints land in
`work_dirs/rtmdet_bag/`. Adjust `configs/rtmdet_bag.py` (batch size, epochs,
LR) if the labeled dataset ends up much bigger/smaller than expected —
comments in the config explain what was tuned down from the base 8-GPU
schedule and why.

**Controlling training quality, while it runs or after:**

- Every `val_interval` epochs (5, by default) the console prints
  `coco/bbox_mAP`, `coco/bbox_mAP_50`, etc. on the held-out val split — that's
  the main signal to watch.
- Full logs land in `work_dirs/rtmdet_bag/<timestamp>/` — `*.log` has the
  human-readable text log (loss per iteration, LR, ETA); `vis_data/scalars.json`
  has the same as newline-delimited JSON if you want to plot loss/mAP curves.
- `default_hooks.checkpoint` uses `save_best='auto'`, so
  `work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth` is the checkpoint to
  use in step 4 — not necessarily the last epoch's.
- If mAP stays near-zero past epoch ~15-20: that's usually a labeling
  problem, not a training one — go back to step 2's `previews/` and check
  recall/false-positive rate before touching hyperparameters.
- If the loss curve is noisy or diverges: lower `optim_wrapper.optimizer.lr`
  in the config and restart (`--resume` picks up from the last checkpoint
  instead of starting over).

## 4. Validate the trained detector

```bash
uv run python scripts/infer.py \
    configs/rtmdet_bag.py \
    work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth \
    --video ../storage/input/input.mp4 \
    --out ../storage/output/rtmdet_preview.mp4
```

Draws per-frame detections on the full video and prints detection-count
stats (mean/min/max per frame, % of frames with zero detections — a quick
signal for missed-detection gaps before we design the tracking/counting
layer on top). This does **not** do cross-frame tracking or deduplicated
belt counting — that algorithm is still an open decision (see
`../app/tracker.py`); this script only validates the detector in isolation.
