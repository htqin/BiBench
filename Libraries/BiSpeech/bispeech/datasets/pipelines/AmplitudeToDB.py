import torchaudio

from ..builder import PIPELINES

@PIPELINES.register_module()
class AmplitudeToDB(object):
    def __init__(self):
        self.transform = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, data):
        data['img'] = self.transform(data['img'])
        return data