import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES

class DfsmnLayer(BaseModule):
    def __init__(self,
                 hidden_size,
                 backbone_memory_size,
                 left_kernel_size,
                 right_kernel_size,
                 dilation=1,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None
    ):
        super(DfsmnLayer, self).__init__(init_cfg=init_cfg)
        self.fc_trans = nn.Sequential(*[
            build_conv_layer(conv_cfg, backbone_memory_size, hidden_size, kernel_size=1),
            build_norm_layer(norm_cfg, hidden_size)[1],
            build_activation_layer(act_cfg),
            DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity(),
            build_conv_layer(conv_cfg, hidden_size, backbone_memory_size, kernel_size=1),
            build_norm_layer(norm_cfg, backbone_memory_size)[1],
            build_activation_layer(act_cfg),
            DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity(),
        ])
        self.memory = nn.Sequential(*[
            build_conv_layer(conv_cfg, backbone_memory_size, backbone_memory_size, kernel_size=left_kernel_size + right_kernel_size + 1,
                      padding=0,
                      stride=1,
                      dilation=dilation,
                      groups=backbone_memory_size),
            build_norm_layer(norm_cfg, backbone_memory_size)[1],
            build_activation_layer(act_cfg),
        ])

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.backbone_memory_size = backbone_memory_size

    def forward(self, input_feat):
        # input (B, N, T)
        residual = input_feat
        # dfsmn-memory
        pad_input_fea = F.pad(input_feat, [
            self.left_kernel_size * self.dilation,
            self.right_kernel_size * self.dilation
        ])  # (B,N,T+(l+r)*d)
        
        memory_out = self.memory(pad_input_fea) + residual
        residual = memory_out  # (B, N, T)
        
        # fc-transform
        fc_output = self.fc_trans(memory_out)  # (B, T, N)
        output = fc_output + residual  # (B, N, T)
        self.layer_output = output
        return output


@BACKBONES.register_module()
class Dfsmn(BaseBackbone):
    def __init__(self,
                 in_channels,
                 n_mels=32,
                 num_layer=6,
                 frondend_channels=16,
                 frondend_kernel_size=5,
                 hidden_size=256,
                 backbone_memory_size=128,
                 left_kernel_size=2,
                 right_kernel_size=2,
                 dilation=1,
                 first_conv_cfg=dict(type='Conv2d'),
                 conv2d_cfg=dict(type='Conv2d'),
                 conv1d_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN2d'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None
        ):
        super(Dfsmn, self).__init__(init_cfg)
        self.front_end = nn.Sequential(*[
            build_conv_layer(first_conv_cfg,
                             in_channels,
                             out_channels=frondend_channels,
                             kernel_size=frondend_kernel_size,
                             stride=(2, 2),
                             padding=(frondend_kernel_size // 2,
                                      frondend_kernel_size // 2)
            ),
            build_norm_layer(norm_cfg, frondend_channels)[1],
            build_activation_layer(act_cfg),
            build_conv_layer(conv2d_cfg,
                             frondend_channels,
                             out_channels=2 * frondend_channels,
                             kernel_size=frondend_kernel_size,
                             stride=(2, 2),
                             padding=(frondend_kernel_size // 2,
                                      frondend_kernel_size // 2)
            ),
            build_norm_layer(norm_cfg, 2 * frondend_channels)[1],
            build_activation_layer(act_cfg),
        ])
        self.n_mels = n_mels
        self.fc1 = nn.Sequential(*[
            nn.Linear(in_features=2 * frondend_channels * self.n_mels // 4,
                      out_features=backbone_memory_size),
            build_activation_layer(act_cfg),
        ])
        backbone = list()
        for idx in range(num_layer):
            backbone.append(
                DfsmnLayer(hidden_size, backbone_memory_size,
                            left_kernel_size, right_kernel_size, dilation,
                            conv_cfg=conv1d_cfg,
                            act_cfg=act_cfg,
                            init_cfg=init_cfg))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, input_feat):
        # input_feat: B, 1, N, T
        batch = input_feat.shape[0]

        out = self.front_end(input_feat)  # B, C, N//4, T//4
        out = out.view(batch, -1, out.shape[3]).transpose(1, 2).contiguous()  # B, T, N1
        out = self.fc1(out).transpose(1, 2).contiguous()  # B, N, T
        for layer in self.backbone:
            out = layer(out)
        
        out = out.contiguous().view(batch, -1)

        return out
