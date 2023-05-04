import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from mmcv.cnn import (build_activation_layer, build_conv_layer)
from mmcv.runner import BaseModule

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES


offset_map = {
    1024: -3.2041,
    2048: -3.4025,
    4096: -3.5836
}


class STN3d(nn.Module):
    def __init__(self,
                 channel,
                 linear_cfg,
                 conv_cfg,
                 act_cfg,
                 pool='max'):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = build_conv_layer(conv_cfg, 64, 128, 1)
        self.conv3 = build_conv_layer(conv_cfg, 128, 1024, 1)
        self.fc1 = build_conv_layer(linear_cfg, 1024, 512)
        self.fc2 = build_conv_layer(linear_cfg, 512, 256)
        self.fc3 = build_conv_layer(linear_cfg, 256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.pool = pool

        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)
        self.activation3 = build_activation_layer(act_cfg)
        self.activation4 = build_activation_layer(act_cfg)

    def forward(self, x):

        batchsize, D, N = x.size()
        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)))

        if self.pool == 'max':
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = self.bn3(self.conv3(x))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.activation3(self.bn4(self.fc1(x)))
        x = self.activation4(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class STNkd(nn.Module):
    def __init__(self, k, linear_cfg, conv_cfg, act_cfg, pool='max'):
        super(STNkd, self).__init__()
        self.conv1 = build_conv_layer(conv_cfg, k, 64, 1)
        self.conv2 = build_conv_layer(conv_cfg, 64, 128, 1)
        self.conv3 = build_conv_layer(conv_cfg, 128, 1024, 1)
        self.fc1 = build_conv_layer(linear_cfg, 1024, 512)
        self.fc2 = build_conv_layer(linear_cfg, 512, 256)
        self.fc3 = build_conv_layer(linear_cfg, 256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k
        self.pool = pool

        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)
        self.activation3 = build_activation_layer(act_cfg)
        self.activation4 = build_activation_layer(act_cfg)

    def forward(self, x):
        batchsize, D, N = x.size()
        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)))
        if self.pool == 'max':
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = self.bn3(self.conv3(x))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = self.activation3(self.bn4(self.fc1(x)))
        x = self.activation4(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self,
                 linear_cfg,
                 conv_cfg,
                 act_cfg,
                 feature_transform=False,
                 channel=3,
                 pool='max',
                 tnet=True):
        super(PointNetEncoder, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = STN3d(channel, linear_cfg, conv_cfg, act_cfg, pool)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = build_conv_layer(conv_cfg, 64, 128, 1)
        self.conv3 = build_conv_layer(conv_cfg, 128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if self.tnet and self.feature_transform:
            self.fstn = STNkd(64, linear_cfg, conv_cfg, act_cfg, pool)
        self.pool = pool
        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)

    def forward(self, x):
        B, D, N = x.size()
        if self.tnet:
            trans = self.stn(x)
        else:
            trans = None

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = self.activation1(self.bn1(self.conv1(x)))

        if self.tnet and self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.activation2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            if self.use_bn:
                x = torch.max(x, 2, keepdim=True)[0] + offset_map[N]
            else:
                x = torch.max(x, 2, keepdim=True)[0] - 0.3
        x = x.view(-1, 1024)
        return x


@BACKBONES.register_module()
class PointNet(BaseModule):
    def __init__(self,
                 normal_channel=True,
                 linear_cfg=dict(type='Linear'),
                 conv_cfg=dict(type='Conv1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 pool='max',
                 tnet=True):
        super(PointNet, self).__init__()
        self.normal_channel = normal_channel
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(
            linear_cfg,
            conv_cfg,
            act_cfg,
            feature_transform=True,
            channel=channel,
            pool=pool,
            tnet=tnet)
        self.fc1 = build_conv_layer(linear_cfg, 1024, 512)
        self.fc2 = build_conv_layer(linear_cfg, 512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation1 = build_activation_layer(act_cfg)
        self.activation2 = build_activation_layer(act_cfg)

    def forward(self, data):
        x = data.permute(0, 2, 1).contiguous()
        if not self.normal_channel:
            x = x[:, :3, :]
        x = self.feat(x)
        x = self.activation1(self.bn1(self.fc1(x)))
        x = self.activation2(self.bn2(self.fc2(x)))
        return x
