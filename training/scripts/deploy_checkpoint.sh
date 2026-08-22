#!/usr/bin/env bash
# Copies the best trained checkpoint + its config into ../models/ and
# commits them, so the running app (and anyone cloning the repo) picks up
# real detections without a manual copy/git dance. Run this after training
# finishes (see ../README.md step 3) on the machine that actually has
# training/work_dirs/ — not on a fresh clone, which has nothing to deploy.
#
# Does not push — review the commit, then `git push` yourself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$TRAINING_DIR")"

CHECKPOINT="$(ls -t "$TRAINING_DIR"/work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)"
if [ -z "$CHECKPOINT" ]; then
    echo "error: no checkpoint found under training/work_dirs/rtmdet_bag/best_coco_bbox_mAP_epoch_*.pth" >&2
    echo "Train first: see training/README.md step 3." >&2
    exit 1
fi

CONFIG="$TRAINING_DIR/configs/rtmdet_bag.py"
if [ ! -f "$CONFIG" ]; then
    echo "error: config not found at $CONFIG" >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/models"
cp "$CONFIG" "$REPO_ROOT/models/rtmdet_bag.py"
cp "$CHECKPOINT" "$REPO_ROOT/models/checkpoint.pth"

echo "Deployed:"
echo "  $CONFIG -> models/rtmdet_bag.py"
echo "  $CHECKPOINT -> models/checkpoint.pth"

cd "$REPO_ROOT"
# git diff --quiet only sees tracked files — a brand-new checkpoint that's
# never been committed is untracked, not "unchanged", so check status
# instead (empty output means nothing new/modified, tracked or not).
if [ -z "$(git status --porcelain -- models/rtmdet_bag.py models/checkpoint.pth)" ]; then
    echo "No changes to commit (checkpoint already matches what's in git)."
    exit 0
fi

git add models/rtmdet_bag.py models/checkpoint.pth
git commit -m "Deploy trained RTMDet checkpoint ($(basename "$CHECKPOINT"))"

echo
echo "Committed. Run 'git push' to publish it — nothing here pushes automatically."
