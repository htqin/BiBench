_base_ = [
    '../_base_/models/resnet18_faster-rcnn.py', '../_base_/datasets/coco.py',
    '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            type='BiBench_ResNet',
            frozen_stages=-1,
            conv_cfg=dict(type='BNNConv'),
            first_act_cfg=dict(type='Hardtanh', inplace=True),
            act_cfg=dict(type='Hardtanh', inplace=True),
            norm_eval=False,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/res18_coco.pth')
)

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)