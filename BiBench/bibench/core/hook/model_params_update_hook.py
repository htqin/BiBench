# Copyright (c) OpenMMLab. All rights reserved
from mmcv.runner import IterBasedRunner, EpochBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
import math
import torch


@HOOKS.register_module()
class ModelParamsUpdateHook(Hook):

    def __init__(self,):
        pass

    def _update_params(self, runner):
        """Update Specific Parameters During Model Training

        Args:
            runner (obj:`EpochBasedRunner`, `IterBasedRunner`): runner object.
            dataset (obj: `BaseDataset`): the dataset to check.
        """
        
        model = runner.model
        for name, module in model.named_modules():
            if hasattr(module, 'model_params_update'):
                module.model_params_update(runner._max_epochs, runner.epoch)

    def before_train_iter(self, runner):
        """

        Args:
            runner (obj: `IterBasedRunner`): Iter based Runner.
        """
        pass

    def before_val_iter(self, runner):
        """

        Args:
            runner (obj:`IterBasedRunner`): Iter based Runner.
        """
        pass

    def before_train_epoch(self, runner):
        """

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._update_params(runner)

    def before_val_epoch(self, runner):
        """

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        pass
