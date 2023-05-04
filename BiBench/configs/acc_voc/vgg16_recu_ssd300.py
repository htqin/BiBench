_base_ = ['./vgg16_bnn_ssd300.py']

extra_hooks = ['ModelParamsUpdateHook']

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='ReCUConv'),
        )
    )
)