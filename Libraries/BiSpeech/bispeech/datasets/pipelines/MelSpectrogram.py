import torchaudio

from ..builder import PIPELINES

@PIPELINES.register_module()
class MelSpectrogram(object):
    def __init__(self, sample_rate=16000,
                       n_fft=2048,
                       hop_length=512,
                       n_mels=32,
                       normalized=True
    ):
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=n_mels,
                                             normalized=normalized),

    def __call__(self, data):
        for t in self.transform:
            data['img'] = t(data['img'])
        return data