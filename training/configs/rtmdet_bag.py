"""RTMDet-tiny fine-tuned for single-class bag detection on conveyor video.

Inherits the COCO-pretrained RTMDet-tiny config bundled with MMDetection
(resolved via the `mmdet::` package-relative syntax, which requires
installing mmdet through `mim install mmdet` rather than plain `pip install`
— mim additionally provisions the `.mim/configs` directory this depends on).
Only the pieces that differ for our dataset/schedule are overridden below;
everything else (backbone, neck, augmentation pipeline, optimizer type) is
inherited as-is from the base config.
"""

_base_ = ["mmdet::rtmdet/rtmdet_tiny_8xb32-300e_coco.py"]

data_root = "data/bags_coco/"
class_name = ("bag",)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train.json",
        data_prefix=dict(img="images/"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/val.json",
        data_prefix=dict(img="images/"),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "annotations/val.json")
test_evaluator = val_evaluator

# Bootstrap-labeled dataset is small (single class, single camera angle) —
# far fewer epochs than the 300-epoch COCO-from-scratch schedule, starting
# from COCO-pretrained weights instead of training from scratch.
#
# stage2_num_epochs mirrors the base config's mosaic/mixup "cooldown" —
# RTMDet trains with heavy Mosaic/MixUp augmentation for most of the
# schedule, then switches to a lighter pipeline (train_pipeline_stage2,
# inherited from the base config) for the last stretch to let the model
# settle. The base config turns this off for its last 20/300 (~7%) epochs;
# we use a larger fraction here since fine-tuning on a small dataset
# benefits from more stabilization time relative to the total run.
max_epochs = 50
stage2_num_epochs = 10
interval = 5

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=200),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0002,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

custom_hooks = [
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0002, update_buffers=True, priority=49),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}},  # noqa: F821 (MMEngine base-config reference syntax)
    ),
]

# Lower than the base config's 0.004 — that schedule assumes 8 GPUs x batch
# 32 (effective batch 256); linear LR scaling for our batch_size=8 on one
# GPU would suggest ~0.004 * 8/256 = 0.000125, but that's overly
# conservative for fine-tuning from a pretrained checkpoint on a small
# dataset, so this is a manually-picked starting point rather than a strict
# linear-scaling result. Watch the loss curve (see training/README.md) and
# adjust if it's noisy (too high) or barely moving (too low).
optim_wrapper = dict(optimizer=dict(lr=0.0004))

load_from = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=interval, max_keep_ckpts=3, save_best="auto"),
    logger=dict(type="LoggerHook", interval=20),
)
