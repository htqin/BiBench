# model settings
arch = dict(
    repo='bipc',
    type='ImageClassifier',
    backbone=dict(
        type='PointNet',
        normal_channel=True,
        linear_cfg=dict(type='Linear'),
        conv_cfg=dict(type='Conv1d'),
        act_cfg=dict(type='ReLU', inplace=True),
        pool='max',
        tnet=True),
    head=dict(
        type='LinearClsHead',
        num_classes=40,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
model = dict(
    type='SimpleArchitecture',
    arch=arch
)