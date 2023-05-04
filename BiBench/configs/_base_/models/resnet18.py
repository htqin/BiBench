# model settings
arch = dict(
    repo='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='BiBench_ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
model = dict(
    type='SimpleArchitecture',
    arch=arch
)