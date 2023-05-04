import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader):
    try:
        repo = model.repo
    except:
        repo = model.module.repo
    if repo == 'mmcls':
        from mmcls.apis import single_gpu_test as task_single_gpu_test
        return task_single_gpu_test(model, data_loader)
    elif repo == 'mmdet':
        from mmdet.apis import single_gpu_test as task_single_gpu_test
        return task_single_gpu_test(model, data_loader)
    elif repo == 'bipc':
        from bipc.apis import single_gpu_test as task_single_gpu_test
        return task_single_gpu_test(model, data_loader)
    elif repo == 'binlp':
        from binlp.apis import single_gpu_test as task_single_gpu_test
        return task_single_gpu_test(model, data_loader)
    elif repo == 'bispeech':
        from bispeech.apis import single_gpu_test as task_single_gpu_test
        return task_single_gpu_test(model, data_loader)
    else:
        raise NotImplementedError()


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    try:
        repo = model.repo
    except:
        repo = model.module.repo
    if repo == 'mmcls':
        from mmcls.apis import multi_gpu_test as task_multi_gpu_test
        return task_multi_gpu_test(model, data_loader, tmpdir, gpu_collect)
    elif repo == 'mmdet':
        from mmdet.apis import multi_gpu_test as task_multi_gpu_test
        return task_multi_gpu_test(model, data_loader, tmpdir, gpu_collect)
    elif repo == 'bipc':
        from bipc.apis import multi_gpu_test as task_multi_gpu_test
        return task_multi_gpu_test(model, data_loader, tmpdir, gpu_collect)
    elif repo == 'binlp':
        from binlp.apis import multi_gpu_test as task_multi_gpu_test
        return task_multi_gpu_test(model, data_loader, tmpdir, gpu_collect)
    elif repo == 'bispeech':
        from bispeech.apis import multi_gpu_test as task_multi_gpu_test
        return task_multi_gpu_test(model, data_loader, tmpdir, gpu_collect)
    else:
        raise NotImplementedError()