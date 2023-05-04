# dataset settings
dataset_repo = 'bipc'
dataset_type='ModelNet'
data_prefix='data/datasets'

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=16,
    train=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix=data_prefix,
        test_mode=False
    ),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix=data_prefix,
        test_mode=True
    ),
    test=dict(
        repo=dataset_repo,
        type=dataset_type,
        data_prefix=data_prefix,
        test_mode=True
    ),
)
