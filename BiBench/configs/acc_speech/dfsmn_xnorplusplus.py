_base_ = [
    '../_base_/models/dfsmn.py', '../_base_/datasets/speech_commands.py',
    '../_base_/schedules/sgd_5e-3_cosinelr_300e_warmup.py', '../_base_/default_runtime.py'
]

model = dict(
    arch=dict(
        backbone=dict(
            conv2d_cfg=dict(type='XNORPlusPlusConv2d'),
            conv1d_cfg=dict(type='XNORPlusPlusConv1d'),
            act_cfg=dict(type='PReLU')
        ),
        init_cfg=dict(type='Pretrained', checkpoint='data/pretrained/dfsmn_speech.pth')
    )
)
