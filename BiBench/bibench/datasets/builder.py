import platform
import random
from functools import partial
from typing import Optional, Union

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import build_from_cfg

from mmcls.datasets import build_dataloader


if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(cfg, default_args=None):
    """"Build dataset by the given config."""
    repo = cfg.pop('repo', None)
    assert repo is not None
    if repo == 'mmcls':
        from mmcls.datasets import build_dataset as build_task_dataset
        dataset = build_task_dataset(cfg, default_args)
    elif repo == 'mmdet':
        from mmdet.datasets import build_dataset as build_task_dataset
        dataset = build_task_dataset(cfg, default_args)
    elif repo == 'bipc':
        from bipc.datasets import build_dataset as build_task_dataset
        dataset = build_task_dataset(cfg, default_args)
    elif repo == 'binlp':
        from binlp.datasets import build_dataset as build_task_dataset
        dataset = build_task_dataset(cfg, default_args)
    elif repo == 'bispeech':
        from bispeech.datasets import build_dataset as build_task_dataset
        dataset = build_task_dataset(cfg, default_args)
    else:
        raise NotImplementedError()
    return dataset
