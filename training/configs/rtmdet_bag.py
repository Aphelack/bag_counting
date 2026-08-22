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
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=100),
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

# Smaller LR than the base config's 8-GPU x batch-32 schedule, since we're
# fine-tuning on one GPU with a much smaller batch size.
optim_wrapper = dict(optimizer=dict(lr=0.0004))

load_from = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=5, max_keep_ckpts=3, save_best="auto"),
    logger=dict(type="LoggerHook", interval=20),
)
