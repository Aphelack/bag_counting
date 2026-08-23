# Training: RTMDet bag detector

Label a dataset from `input.mp4` with SAM 3, fine-tune RTMDet-tiny,
validate, deploy to the app. Runs on a CUDA host — written on a machine
with no GPU, nothing here runs there.

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

> Every time you run `uv sync` after this, redo the three `uv pip` lines
> too. `mmengine`/`mmcv`/`mmdet` are installed via `mim`, invisible to
> uv's lockfile, so `uv sync` treats them as extraneous and removes them
> (`ModuleNotFoundError: No module named 'mmdet'`). `opencv-python` and
> `numpy>=2.0` creep back the same way.

`torch`/`torchvision` are pinned to 2.1.2/cu121 — OpenMMLab's prebuilt
`mmcv` wheels only go up to ~torch 2.7/cu128. Unpinned torch → `mim
install mmcv` finds no wheel → falls back to building from source → fails
separately (`pkg_resources` removed from `setuptools` 81+, hence
`setuptools<81`). If `mim install mmcv` can't find a wheel, check
OpenMMLab's current matrix and adjust the pin in `pyproject.toml`,
`[tool.uv.sources]`, and the `mim install` command together.

Labeling also needs the `sam3` package, in whichever env you're running
`../experiments/notebooks/01_sam3_segmentation_test.ipynb`:

```bash
uv pip install git+https://github.com/facebookresearch/sam3.git
```

## 2. Label a dataset

```bash
uv run --project ../experiments python ../experiments/scripts/label_with_sam3.py \
    --video ../storage/input/input.mp4 \
    --out-dir data/bags_coco \
    --checkpoint-path /path/to/sam3.pt \
    --bpe-path /path/to/bpe_simple_vocab_16e6.txt.gz \
    --stride 25 \
    --confidence-threshold 0.75
```

Check `data/bags_coco/previews/` before training (every 20th labeled
frame, boxes drawn). Bad frames → delete the image + its annotation from
the COCO JSON rather than re-running the pipeline.

## 3. Train

```bash
uv run python scripts/train.py configs/rtmdet_bag.py
```

50 epochs from COCO-pretrained weights. Checkpoints in
`work_dirs/rtmdet_bag/`; `best_coco_bbox_mAP_epoch_*.pth` is the one to
use, not necessarily the last epoch. mAP stuck near-zero → labeling
problem, check the previews first. High val mAP isn't proof of quality —
train/val labels both come from SAM3, so it mostly measures agreement
with SAM3, not ground truth (`notebooks/01_inspect_predictions.ipynb` is
the real check, step 5).

## 4. Validate

```bash
uv run python scripts/infer.py \
    configs/rtmdet_bag.py \
    work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth \
    --video ../storage/input/input.mp4 \
    --out ../storage/output/rtmdet_preview.mp4
```

Detector only, no tracking/counting (that's `../app/tracker.py`).

## 5. Check on unseen frames

```bash
uv sync
uv run python -m ipykernel install --user --name bag-counting-training --display-name "bag-counting-training"
```

Open `notebooks/01_inspect_predictions.ipynb`. Samples frames offset from
the labeling stride (never in train or val), draws predictions — the
actual generalization check.

## 6. Deploy to the app

```bash
./scripts/deploy_checkpoint.sh
git push
```

Copies the best checkpoint + config into `../models/`, commits if
changed. Doesn't push itself.
