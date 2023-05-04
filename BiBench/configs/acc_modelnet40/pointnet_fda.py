_base_ = [
    '../_base_/models/pointnet.py', '../_base_/datasets/modelnet40.py',
    '../_base_/schedules/adam_1e-3_cosinelr_200e.py', '../_base_/default_runtime.py'
]

extra_hooks = ['ModelParamsUpdateHook']

model = dict(
    arch=dict(
        backbone=dict(
            linear_cfg=dict(type='FDALinear'),
            conv_cfg=dict(type='FDAConv1d'),
            act_cfg=dict(type='Hardtanh')
        )
    )
)
