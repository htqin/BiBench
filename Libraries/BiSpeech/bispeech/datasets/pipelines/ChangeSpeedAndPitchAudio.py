import random
import torch

from ..builder import PIPELINES

@PIPELINES.register_module()
class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""
    def __init__(self, prop=0.5, max_scale=0.2, sample_rate=16000):
        self.max_scale = max_scale
        self.sample_rate = sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            data['img'] = torch.nn.functional.interpolate(data['img'].unsqueeze(1),
                                                   scale_factor=speed_fac,
                                                   mode='nearest').squeeze(1)
        return data
