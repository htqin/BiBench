_base_ = ['./resnet18_bnn_faster-rcnn.py']

model = dict(
    arch=dict(
        backbone=dict(
            conv_cfg=dict(type='XNORConv'),
        )
    )
)