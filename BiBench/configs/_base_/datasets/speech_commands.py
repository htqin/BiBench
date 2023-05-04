# dataset settings
dataset_repo = 'bispeech'
dataset_type='SpeechCommandDataset'
n_mels=32
data_prefix='data/datasets'
num_classes=12
version="speech_commands_v0.01"

train_pipeline = [
    dict(type='ChangeAmplitude'),
    dict(type='ChangeSpeedAndPitchAudio'),
    dict(type='TimeshiftAudio'),
    dict(type='FixAudioLength'),
    dict(type='MelSpectrogram', sample_rate=16000, n_fft=2048, hop_length=512, n_mels=n_mels, normalized=True),
    dict(type='AmplitudeToDB'),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='FixAudioLength'),
    dict(type='MelSpectrogram', sample_rate=16000, n_fft=2048, hop_length=512, n_mels=n_mels, normalized=True),
    dict(type='AmplitudeToDB'),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=96,
    workers_per_gpu=16,
    train=dict(
        repo=dataset_repo,
        type=dataset_type,
        subset='training',
        data_prefix=data_prefix,
        #download=True,
        noise_ratio=0.3,
        noise_max_scale=0.3,
        num_classes=num_classes,
        version=version,
        pipeline=train_pipeline
    ),
    val=dict(
        repo=dataset_repo,
        type=dataset_type,
        subset='validation',
        data_prefix=data_prefix,
        #download=True,
        num_classes=num_classes,
        version=version,
        pipeline=test_pipeline
    ),
    test=dict(
        repo=dataset_repo,
        type=dataset_type,
        subset='testing',
        data_prefix=data_prefix,
        #download=True,
        num_classes=num_classes,
        version=version,
        pipeline=test_pipeline
    ),
)
