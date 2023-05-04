_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet.py',
    '../_base_/schedules/adam_1e-3_cosinelr_e100.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='BNNConv'),
            first_act_cfg=dict(type='Hardtanh', inplace=True),
            act_cfg=dict(type='Hardtanh', inplace=True),
            init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/res18_imagenet.pth')
        )
    )
)
