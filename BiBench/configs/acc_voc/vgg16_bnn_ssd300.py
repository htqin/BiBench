_base_ = [
    './vgg16_fp_ssd300.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            type='BiBench_SSDVGG',
            conv_cfg=dict(type='BNNConv'),
            act_cfg=dict(type='Hardtanh', inplace=True)
        )
    ),
    init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/vgg_voc.pth')
)

data = dict(samples_per_gpu=32)

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)