# model_pipeline/configs/hrnet_w32_horse68_256x256.py
# HRNet-W32 top-down pose estimation for 68 horse keypoints (COCO-style dataset)
import os
import time

# Skapa en unik run-id för den här körningen (YYMMDD_HHMM)
run_id = time.strftime("%y%m%d_%H%M")  # t.ex. 250910_1420
work_dir = os.path.join('outputs/checkpoints', f'horse_hrnet_{run_id}')



# ===== Data paths =====
data_root       = 'dataset_pipeline/data/dataset/final'   # Rot till bilderna
train_ann_file  = 'coco_files/coco_synth_68.json'
val_ann_file    = 'coco_files/coco_synth_68.json'
metainfo=dict(from_file='model_pipeline/configs/_base_/horse68.py')

# ===== Codec / Heatmap settings =====
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2.0
)

# ===== Model =====
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
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK',
                        num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC',
                        num_blocks=(4, 4), num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC',
                        num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC',
                        num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256)),
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=68,
        deconv_out_channels=None,
        loss=dict(
            type='KeypointMSELoss',
            use_target_weight=True,
            loss_weight=1.0
        ),
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
    dict(type='RandomHalfBody', prob=0.3),
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
        metainfo=dict(from_file='model_pipeline/configs/_base_/horse68.py'),
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
        metainfo=dict(from_file='model_pipeline/configs/_base_/horse68.py'),
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
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
        metainfo=dict(from_file='model_pipeline/configs/_base_/horse68.py'),
        pipeline=val_pipeline
    )
)

# ===== Evaluators =====
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(data_root, val_ann_file)
)

test_evaluator = val_evaluator

# ===== Optimization / schedule =====
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=1e-4)
)

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[100, 140], gamma=0.1)
]

# train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=5) # <-- Kör över natten
train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1) # <-- Kör under dagen
val_cfg = dict()
test_cfg = dict()

# ===== Hooks / logging =====
default_scope = 'mmpose'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5, save_best='coco/AP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50)
)

env_cfg = dict(cudnn_benchmark=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
load_from = None   # optional: path to pretrained HRNet-W32 weights
resume = False
