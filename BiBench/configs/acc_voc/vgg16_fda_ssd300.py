_base_ = ['./vgg16_bnn_ssd300.py']

extra_hooks = ['ModelParamsUpdateHook']

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='FDAConv'),
        )
    )
)

data = dict(samples_per_gpu=8)