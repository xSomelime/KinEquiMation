# configs/hrnet_w32_horse68_256x256.py
# Minimal HRNet-W32 top-down heatmap config för 68 häst-nyckelpunkter.
# Byt sökvägar under "DATA PATHS" nedan till dina faktiska filer.

backend_args = None

# ===== DATA PATHS =====
train_ann_file = 'ai/data/coco_synth_68.json'   # från synth_to_coco.py
val_ann_file   = 'ai/data/coco_synth_68.json'   # (ev. egen val-fil, men ok för PoC)
# Bildfilerna refereras med absoluta paths i annotations -> ok att sätta data_root=''

# ===== Metainfo loader (läser din skeleton_edges + def_bones) =====
import json, os
def load_metainfo(def_bones_path='ai/data/def_bones.txt',
                  skeleton_json='ai/data/skeleton_edges.json'):
    with open(def_bones_path, 'r', encoding='utf-8') as f:
        kps = [ln.strip() for ln in f if ln.strip()]
    name_to_idx = {n:i for i,n in enumerate(kps)}
    with open(skeleton_json, 'r', encoding='utf-8') as f:
        edges = json.load(f).get('edges', [])
    # COCO skeleton är 1-baserad indexering
    skeleton = []
    for pa, ch in edges:
        if pa in name_to_idx and ch in name_to_idx:
            skeleton.append([name_to_idx[pa]+1, name_to_idx[ch]+1])
    return dict(from_file=None, dataset_name='horse68',
                keypoint_names=kps, keypoint_colors=None,
                skeleton=skeleton, skeleton_links=None)

metainfo = load_metainfo()

# ===== Model =====
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
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK',
                        num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC',
                        num_blocks=(4,4), num_channels=(32,64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC',
                        num_blocks=(4,4,4), num_channels=(32,64,128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC',
                        num_blocks=(4,4,4,4), num_channels=(32,64,128,256)),
        )),
    head=dict(
        type='HeatmapHead',    # enkel 1x1 head
        in_channels=32,        # HRNet-W32 lågupplöst gren → 32 kanaler
        out_channels=len(metainfo['keypoint_names']),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=None),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True)
)

# ===== Codec / Heatmap-inställningar =====
codec = dict(type='MSRAHeatmap', input_size=(256,256), heatmap_size=(64,64), sigma=2.0)

# ===== Pipelines =====
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='RandomHalfBody', num_joints_half_body=len(metainfo['keypoint_names'])//2, prob=0.3),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256,256)),
    dict(type='PackPoseInputs')
]

# ===== Dataloaders =====
train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file=train_ann_file,
        data_mode='topdown',
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16, num_workers=2, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file=val_ann_file,
        data_mode='topdown',
        test_mode=True,
        metainfo=metainfo,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

# ===== Optimering / schema =====
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[100, 140], gamma=0.1)
]
train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# ===== Hookar / logg =====
default_scope = 'mmpose'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=5, save_best='PCK@0.2'),
    logger=dict(type='LoggerHook', interval=50))
env_cfg = dict(cudnn_benchmark=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_level = 'INFO'
load_from = None  # (valfritt) lägg in förtränade HRNet-W32-vikter här om du vill
resume = False
