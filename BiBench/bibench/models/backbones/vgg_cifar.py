# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer

from ..builder import BACKBONES
from .vgg import VGG