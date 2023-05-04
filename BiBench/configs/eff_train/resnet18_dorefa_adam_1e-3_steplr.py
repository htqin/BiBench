_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/cifar10.py',
    '../_base_/schedules/adam_1e-3_steplr_e200.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            type='BiBench_ResNet_CIFAR',
            conv_cfg=dict(type='DoReFaConv'),
            first_act_cfg=dict(type='Hardtanh', inplace=True),
            act_cfg=dict(type='Hardtanh', inplace=True)
        ),
        head=dict(num_classes=10)
    )
)
