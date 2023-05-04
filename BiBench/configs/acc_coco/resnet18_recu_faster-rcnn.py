_base_ = ['./resnet18_bnn_faster-rcnn.py']

extra_hooks = ['ModelParamsUpdateHook']

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='ReCUConv'),
        )
    )
)