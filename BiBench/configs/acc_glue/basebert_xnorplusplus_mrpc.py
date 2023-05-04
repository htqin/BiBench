_base_ = [
    '../_base_/models/basebert.py', '../_base_/datasets/glue_mrpc.py',
    '../_base_/schedules/glue_mrpc.py', '../_base_/default_runtime.py'
]

pretrained_path = 'data/pretrained/dynabert/MRPC'
model = dict(
    arch=dict(
        teacher=dict(pretrained_path=pretrained_path, num_labels=2),
        student=dict(
            pretrained_path=pretrained_path,
            num_labels=2,
            linear_cfg=dict(type='XNORPlusPlusLinear')
        )
    )
)
checkpoint_config = dict(interval=200, save_last=True)