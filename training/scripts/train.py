"""Launch RTMDet training from an MMDetection config.

Equivalent to mmdetection's `tools/train.py`, but self-contained — no need to
clone the full mmdetection repo, just `mim install mmdet` (see
../README.md). Not runnable in this dev environment (no GPU, no mmdet
installed); meant to run on the GPU host after `label_with_sam3.py` has
produced `training/data/bags_coco/`.

Usage:
    uv run python scripts/train.py configs/rtmdet_bag.py
"""
from __future__ import annotations

import argparse

from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to MMDetection config, e.g. configs/rtmdet_bag.py")
    parser.add_argument("--work-dir", default=None, help="override the config's work_dir")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in work_dir")
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.resume = args.resume

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
