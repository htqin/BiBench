# optimizer
optimizer = dict(type='SGD', lr=1e-1, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=400)