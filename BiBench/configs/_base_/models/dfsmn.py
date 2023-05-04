# model settings
arch = dict(
    repo='bispeech',
    type='ImageClassifier',
    backbone=dict(
        type='Dfsmn',
        in_channels=1,
        n_mels=32,
        num_layer=8,
        frondend_channels=16,
        frondend_kernel_size=5,
        hidden_size=256,
        backbone_memory_size=128,
        left_kernel_size=2,
        right_kernel_size=2,
        dilation=1,
        drop_path_rate=0.0,
    ),
    head=dict(
        type='LinearClsHead',
        in_channels=1024, # in_channels = backbone_memory_size * n_mels // 4
        num_classes=12,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
model = dict(
    type='SimpleArchitecture',
    arch=arch
)