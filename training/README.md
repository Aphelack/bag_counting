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
# mim installs via plain pip internally, which ignores pyproject.toml's
# pins entirely and will happily break both of these:
# 1. mmcv/mmdet pull in plain opencv-python alongside our
#    opencv-python-headless — having both breaks cv2 (AttributeError:
#    module 'cv2' has no attribute 'VideoCapture').
# 2. mmcv/mmdet's own deps drag numpy up to 2.x, but our pinned
#    torch==2.1.2 was compiled against NumPy 1.x — mismatched ABI breaks
#    torch/numpy interop (RuntimeError: Numpy is not available, or a
#    "compiled using NumPy 1.x cannot be run in NumPy 2.x" warning first).
# Fix both right after every mim install:
uv pip uninstall opencv-python
uv pip install --reinstall --no-deps opencv-python-headless
uv pip install "numpy<2.0"
```

> **Every time you run `uv sync` after this, redo the four lines below it
> again too.** `mmengine`/`mmcv`/`mmdet` are installed via `mim`, not
> declared in `pyproject.toml`, so they're invisible to uv's lockfile —
> `uv sync` treats them as extraneous and silently removes them, which
> shows up later as `ModuleNotFoundError: No module named 'mmdet'` with no
> obvious cause. `opencv-python` and `numpy>=2.0` creeping back in after a
> fresh `mim install` need the same cleanup pass each time too.

`torch`/`torchvision` are pinned to 2.1.2/cu121 in `pyproject.toml` (not left
open) because OpenMMLab's prebuilt `mmcv` wheels only go up to around torch
2.7/cu128 as of writing — mmcv development has slowed a lot (see
`../experiments/README.md`'s MMDetection maintenance note). An unpinned
`torch` resolves to whatever's newest, `mim install mmcv` can't find a
matching wheel for that, and falls back to building mmcv from source, which
fails for unrelated reasons (its legacy `setup.py` needs `pkg_resources`,
removed from `setuptools` 81+ — hence the `setuptools<81` pin too). A CUDA
12.1 *runtime* wheel works fine on newer GPU drivers (driver forward
compatibility), so this pin shouldn't need a matching CUDA *toolkit*
install — but if `mim install mmcv` still can't find a wheel, check
OpenMMLab's current supported matrix and adjust the pin here, in
`[tool.uv.sources]` below, and in the `mim install` command's `mmcv`
constraint together.

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

**A very high val mAP here is not proof the detector is good** — train and
val labels both came from the same SAM3 pass, so this metric measures how
well RTMDet reproduced SAM3's boxes, including any systematic labeling
mistakes SAM3 made (e.g. the floor bag-pile / laundry confusion from
earlier testing). Treat it as a training-convergence signal, not a
real-world accuracy number — `notebooks/01_inspect_predictions.ipynb`
(step 5 below) is the actual check.

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

## 5. Eyeball predictions on frames the model never saw

```bash
uv sync   # picks up jupyter/matplotlib, added for this notebook
uv run python -m ipykernel install --user --name bag-counting-training --display-name "bag-counting-training"
```

Open `notebooks/01_inspect_predictions.ipynb`, kernel `bag-counting-training`.
Samples frames offset from the labeling `--stride` grid (so none of them were
in train or val), runs the trained checkpoint on them, and draws the boxes —
this is the actual check for whether the detector generalizes, versus just
agreeing with SAM3's own labels (see the mAP caveat in step 3).
