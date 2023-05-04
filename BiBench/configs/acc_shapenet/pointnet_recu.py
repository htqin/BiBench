_base_ = [
    '../_base_/models/pointnet_seg.py', '../_base_/datasets/shapenet.py',
    '../_base_/schedules/adam_1e-3_cosinelr_200e.py', '../_base_/default_runtime.py'
]

extra_hooks = ['ModelParamsUpdateHook']

model = dict(
    arch=dict(
        backbone=dict(
            linear_cfg=dict(type='ReCULinear'),
            conv_cfg=dict(type='ReCUConv1d'),
            act_cfg=dict(type='Hardtanh')
        )
    )
)
