# model settings
arch = dict(
    repo='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='BiBench_ReActNet',
        conv_cfg=dict(type='Conv2d')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    )
)
model = dict(
    type='SimpleArchitecture',
    arch=arch
)