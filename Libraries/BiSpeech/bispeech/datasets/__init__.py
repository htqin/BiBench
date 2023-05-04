# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.datasets import *

from .pipelines import *
from .speech_commands import SpeechCommandDataset

__all__ = [
    'SpeechCommandDataset'
]
