_base_ = [
    '../_base_/models/basebert.py', '../_base_/datasets/glue_cola.py',
    '../_base_/schedules/glue_cola.py', '../_base_/default_runtime.py'
]

pretrained_path = 'data/pretrained/dynabert/CoLA'
model = dict(
    arch=dict(
        teacher=dict(pretrained_path=pretrained_path, num_labels=2),
        student=dict(
            pretrained_path=pretrained_path,
            num_labels=2,
            linear_cfg=dict(type='ReActLinear')
        )
    )
)
checkpoint_config = dict(interval=500, save_last=True)