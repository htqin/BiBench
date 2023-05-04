# dataset settings
dataset_repo = 'mmcls'
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=16,
    persistent_workers=True,
    train=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix='data/datasets/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix='data/datasets/imagenet/val',
        ann_file='data/datasets/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix='data/datasets/imagenet/val',
        ann_file='data/datasets/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy', save_best='accuracy_top-1')