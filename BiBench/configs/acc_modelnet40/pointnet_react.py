_base_ = [
    '../_base_/models/pointnet.py', '../_base_/datasets/modelnet40.py',
    '../_base_/schedules/adam_1e-3_cosinelr_200e.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            linear_cfg=dict(type='ReActLinear'),
            conv_cfg=dict(type='ReActConv1d'),
            act_cfg=dict(type='Hardtanh')
        )
    )
)
