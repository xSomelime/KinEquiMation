from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *


# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy# model_pipeline/configs/hrnet_w32_horse68_256x256.py
# HRNet-W32 top-down pose config for 68 horse keypoints (COCO format)

import json
import os

backend_args = None

# ===== DATA PATHS =====
train_ann_file = 'dataset_pipeline/data/dataset_exports/coco_synth_68.json'
val_ann_file   = 'dataset_pipeline/data/dataset_exports/coco_synth_68.json'

# ===== Metainfo loader (reads skeleton + keypoints) =====
def load_metainfo(
    def_bones_path='dataset_pipeline/data/dataset_exports/def_bones.txt',
    skeleton_json='dataset_pipeline/data/dataset_exports/skeleton_edges.json'
):
    with open(def_bones_path, 'r', encoding='utf-8') as f:
        kps = [ln.strip() for ln in f if ln.strip()]
    name_to_idx = {n: i for i, n in enumerate(kps)}
    with open(skeleton_json, 'r', encoding='utf-8') as f:
        edges = json.load(f).get('edges', [])
    skeleton = []
    for pa, ch in edges:
        if pa in name_to_idx and ch in name_to_idx:
            skeleton.append([name_to_idx[pa] + 1, name_to_idx[ch] + 1])
    return dict(
        from_file=None,
        dataset_name='horse68',
        keypoint_names=kps,
        keypoint_colors=None,
        skeleton=skeleton,
        skeleton_links=None
    )

metainfo = load_metainfo()

# ===== Model =====
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2.0)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1, num_branches=1, block='BOTTLENECK',
                num_blocks=(4,), num_channels=(64,)
            ),
            stage2=dict(
                num_modules=1, num_branches=2, block='BASIC',
                num_blocks=(4, 4), num_channels=(32, 64)
            ),
            stage3=dict(
                num_modules=4, num_branches=3, block='BASIC',
                num_blocks=(4, 4, 4), num_channels=(32, 64, 128)
            ),
            stage4=dict(
                num_modules=3, num_branches=4, block='BASIC',
                num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256)
            )
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=len(metainfo['keypoint_names']),
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True
    )
)

# ===== Pipelines =====
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomHalfBody', num_joints_half_body=len(metainfo['keypoint_names']) // 2, prob=0.3),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# ===== Dataloaders =====
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = ''

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_mode=data_mode,
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_mode=data_mode,
        test_mode=True,
        metainfo=metainfo,
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# ===== Optimization / schedule =====
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[100, 140], gamma=0.1)
]
train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# ===== Hooks / logging =====
default_scope = 'mmpose'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5, save_best='PCK@0.2'),
    logger=dict(type='LoggerHook', interval=50)
)
env_cfg = dict(cudnn_benchmark=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_level = 'INFO'
load_from = None   # optional: path to pretrained weights if you want to warm start
resume = False

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=20,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'AnimalPoseDataset'
data_mode = 'topdown'
data_root = 'data/animalpose/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/animalpose_train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/animalpose_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/animalpose_val.json')
test_evaluator = val_evaluator
