# model settings
arch = dict(
    repo='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='BiBench_ResNet',
        depth=20,
        base_channels=16,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
model = dict(
    type='SimpleArchitecture',
    arch=arch
)
