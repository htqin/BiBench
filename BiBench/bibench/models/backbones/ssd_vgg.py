# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn import build_activation_layer, build_conv_layer

import logging
from typing import List, Optional, Sequence, Tuple, Union

from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, conv_cfg: dict, dilation: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return build_conv_layer(
        conv_cfg,
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


def make_vgg_layer(inplanes: int,
                   planes: int,
                   num_blocks: int,
                   conv_cfg: dict,
                   act_cfg: dict,
                   dilation: int = 1,
                   with_bn: bool = False,
                   ceil_mode: bool = False) -> List[nn.Module]:
    layers = []
    for i in range(num_blocks):
        if i == 0 and inplanes == 3:
            layers.append(conv3x3(inplanes, planes, None, dilation))
        else:
            layers.append(conv3x3(inplanes, planes, conv_cfg, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(build_activation_layer(act_cfg))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers



class SSDVGG(BaseModule):
    """VGG Backbone network for single-shot-detection.
    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_last_pool (bool): Whether to add a pooling layer at the last
            of the model
        ceil_mode (bool): When True, will use `ceil` instead of `floor`
            to compute the output shape.
        out_indices (Sequence[int]): Output from which stages.
        out_feature_indices (Sequence[int]): Output from which feature map.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
        input_size (int, optional): Deprecated argumment.
            Width and height of input, from {300, 512}.
        l2_norm_scale (float, optional) : Deprecated argumment.
            L2 normalization layer init scale.
    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 num_stages: int = 5,
                 frozen_stages=-1,
                 dilations: Sequence[int] = (1, 1, 1, 1, 1),
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 bn_eval=False,
                 bn_frozen=False,
                 with_bn: bool = False,
                 pretrained=None,
                 init_cfg=None,
                 input_size=None,
                 l2_norm_scale=None,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(SSDVGG, self).__init__(init_cfg=init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.inplanes,
                planes,
                num_blocks,
                conv_cfg,
                act_cfg,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'

        if init_cfg is not None:
            self.init_cfg = init_cfg
        elif isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]
        else:
            raise TypeError('pretrained must be a str or None')

        if input_size is not None:
            warnings.warn('DeprecationWarning: input_size is deprecated')
        if l2_norm_scale is not None:
            warnings.warn('DeprecationWarning: l2_norm_scale in VGG is '
                          'deprecated, it has been moved to SSDNeck.')

    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False

