_base_ = [
    '../_base_/models/reactnet.py', '../_base_/datasets/cifar10.py',
    '../_base_/schedules/adam_1e-3_cosinelr_400e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='ReActConv'),
        ),
        head=dict(num_classes=10)
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/reactnet_cifar10.pth')
)
