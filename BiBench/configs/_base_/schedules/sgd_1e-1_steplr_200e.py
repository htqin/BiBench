# optimizer
optimizer = dict(type='SGD', lr=1e-1, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)