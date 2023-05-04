# optimizer
max_iters = 60000
optimizer = dict(
    type='BertAdam',
    schedule='warmup_linear',
    lr=5e-5,
    warmup=0.1,
    t_total=max_iters)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
runner = dict(type='IterBasedRunner', max_iters=max_iters)