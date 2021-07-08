"""Fast Video dataloader for MVFNet R50
"""
import datetime
import os

import numpy as np

T = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

## input settings
# 【option】: 4x16, 8x8, 16x4
clip_len = 4
frame_interval = 16


model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='pretrained/resnet50.pth',
        depth=50,
        out_indices=(3,),
        norm_eval=False,
        partial_norm=False,
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    cls_head=dict(
        type='TSNClsHead',
        spatial_size=-1,
        spatial_type='avg',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        init_std=0.01,
        num_classes=400),
    module_cfg=dict(
        type='MVF',
        n_segment=clip_len,
        alpha=0.125,
        mvf_freq=(0, 0, 1, 1),
        mode='THW')
)


train_cfg = None
test_cfg = None
# dataset settings
root = '/data/'
dataset_type = 'VideoDataset'
data_root = root + 'k400_transcode_video/train_video'
data_root_val = root + 'k400_transcode_video/val_video'
ann_file_train = 'datalist/kinetics400/video_train.txt'
ann_file_val = 'datalist/kinetics400/video_val.txt'
ann_file_test = 'datalist/kinetics400/video_val.txt'


train_pipeline = [
    dict(type='SampleFrames',
         clip_len=clip_len,
         frame_interval=frame_interval,
         num_clips=1),
    dict(type='PyAVDecode', accurate=False),
    dict(
        type='RandomResizedCrop',
        input_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['img_group', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['img_group', 'label']),
]

val_pipeline = [
    dict(type='SampleFrames',
         clip_len=clip_len,
         frame_interval=frame_interval,
         num_clips=1),
    dict(type='PyAVDecode', accurate=False),
    dict(type='Resize', scale=(np.Inf, 256), keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['img_group', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['img_group']),
]

test_pipeline = [
    dict(type='SampleFrames',
         clip_len=clip_len,
         frame_interval=frame_interval,
         num_clips=10),
    dict(type='PyAVDecode', accurate=False),
    dict(type='Resize', scale=(np.Inf, 256), keep_ratio=True),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['img_group', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['img_group']),
]

data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_root=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_root=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9,
                 weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='step',
    step=[90, 130],
    warmup_ratio=0.01,
    warmup='linear',
    warmup_iters=25070)
checkpoint_config = dict(interval=10)
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments/mvfnet/kinetics400_2d_video_r50_%dx%d_dense_b12_g8' % (
    clip_len, frame_interval)
load_from = None
resume_file = os.path.join(work_dir, 'latest.pth')
resume_from = resume_file if os.path.exists(resume_file) else None
eval_interval = 10
cudnn_benchmark = True
# fp16 settings
# fp16 = dict(loss_scale=512.)
