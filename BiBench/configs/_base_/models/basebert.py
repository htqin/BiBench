# model settings
arch = dict(
    repo='binlp',
    type='DistillationClassifier',
    teacher=dict(
        type='BertSeqClassifier',
        pretrained_path='data/pretrained/dynabert/MNLI',
        num_labels=3,
        linear_cfg=dict(type='Linear'),
    ),
    student=dict(
        type='BertSeqClassifier',
        pretrained_path='data/pretrained/dynabert/MNLI',
        num_labels=3,
        linear_cfg=dict(type='Linear'),
    )
)
model = dict(
    type='SimpleArchitecture',
    arch=arch
)