# dataset settings
dataset_repo = 'binlp'
dataset_type = 'GLUEDataset'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        repo=dataset_repo,
        type=dataset_type,
        task='cola',
        data_prefix='data/datasets/GLUE/cola',
        max_seq_length=64,
        tokenizer_path='data/pretrained/dynabert/CoLA'),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        task='cola',
        data_prefix='data/datasets/GLUE/cola',
        max_seq_length=64,
        tokenizer_path='data/pretrained/dynabert/CoLA',
        test_mode=True))
evaluation = dict(interval=500)