import random
import torch

from ..builder import PIPELINES

@PIPELINES.register_module()
class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""
    def __init__(self, prop=0.5, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range
        self.prop = prop

    def __call__(self, data: torch.Tensor):
        if random.uniform(0, 1) <= self.prop:
            data['img'] = data['img'] * random.uniform(*self.amplitude_range)
        return data
