# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.01, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True
)
runner = dict(type='EpochBasedRunner', max_epochs=300)
