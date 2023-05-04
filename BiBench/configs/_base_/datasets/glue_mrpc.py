# dataset settings
dataset_repo = 'binlp'
dataset_type = 'GLUEDataset'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        repo=dataset_repo,
        type=dataset_type,
        task='mrpc',
        data_prefix='data/datasets/GLUE/mrpc',
        max_seq_length=128,
        tokenizer_path='data/pretrained/dynabert/MRPC'),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        task='mrpc',
        data_prefix='data/datasets/GLUE/mrpc',
        max_seq_length=128,
        tokenizer_path='data/pretrained/dynabert/MRPC',
        test_mode=True))
evaluation = dict(interval=200)