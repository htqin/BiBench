import random
import torch

from ..builder import PIPELINES

@PIPELINES.register_module()
class TimeshiftAudio(object):
    """Shifts an audio randomly."""
    def __init__(self, prop=0.5, max_shift_seconds=0.2, sample_rate=16000):
        self.shift_len = max_shift_seconds * sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            shift = random.randint(-self.shift_len, self.shift_len)
            a = -min(0, shift)
            b = max(0, shift)
            data['img'] = torch.nn.functional.pad(data['img'], (a, b), "constant")
            data['img'] = data['img'][:, :data['img'].shape[1] - a] if a else data['img'][:, b:]
        return data
