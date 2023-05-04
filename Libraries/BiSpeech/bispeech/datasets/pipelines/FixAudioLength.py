import torch

from ..builder import PIPELINES

@PIPELINES.register_module()
class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""
    def __init__(self, time=1, sample_rate=16000):
        self.target_len = time * sample_rate

    def __call__(self, data: torch.Tensor):
        cur_len = data['img'].shape[1]
        if self.target_len <= cur_len:
            data['img'] = data['img'][:, :self.target_len]
        else:
            data['img'] = torch.nn.functional.pad(data['img'], (0, self.target_len - cur_len))
        return data