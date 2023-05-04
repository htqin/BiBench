_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/cifar10.py',
    '../_base_/schedules/adam_1e-3_cosinelr_400e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            type='BiBench_ResNet_CIFAR',
            conv_cfg=dict(type='BiRealConv'),
            first_act_cfg=dict(type='Hardtanh', inplace=True),
            act_cfg=dict(type='Hardtanh', inplace=True)
        ),
        head=dict(num_classes=10)
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/res18_cifar10.pth')
)
