from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule
from .utils import build_arch


class SimpleArchitecture(BaseModule):
    """Simple architecture."""

    def __init__(self, arch, init_cfg=None):
        super(SimpleArchitecture, self).__init__(init_cfg)
        self.repo = arch['repo']
        self.arch = build_arch(arch)

    def train_step(self, *args, **kwargs):
        return self.arch.train_step(*args, **kwargs)

    def val_step(self, *args, **kwargs):
        return self.arch.val_step(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.arch.forward(*args, **kwargs)
