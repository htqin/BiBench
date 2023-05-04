# dataset settings
dataset_repo = 'mmcls'
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        repo=dataset_repo,
        type=dataset_type, 
        data_prefix='data/datasets/cifar10',
        pipeline=train_pipeline),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix='data/datasets/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix='data/datasets/cifar10',
        pipeline=test_pipeline,
        test_mode=True))
evaluation=dict(save_best='accuracy_top-1')
