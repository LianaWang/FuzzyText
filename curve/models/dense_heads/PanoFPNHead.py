import torch
from torch import nn
from .StuffHeadBase import StuffHeadBase
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
from mmdet.models import HEADS
from mmcv.runner import auto_fp16, force_fp32


__all__ = ['PanoFpnHead']


class FCNSubNet(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 num_layers=1,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FCNSubNet, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            conv.append(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    inner_channels,
                    kernel_size=3,
                    stride=1,
                    # not using dilated conv
                    padding=1,
                    dilation=1,
                )
            )
            in_channels = inner_channels
            if norm_cfg is not None:
                norm_name, norm_layer = build_norm_layer(norm_cfg, in_channels, postfix=i)
                conv.append(norm_layer)
            conv.append(nn.ReLU(inplace=True))
            conv.append(nn.Upsample(scale_factor=2.0, align_corners=False, mode='bilinear'))
            self.convs.append(nn.Sequential(*conv))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x


@HEADS.register_module()
class PanoFpnHead(StuffHeadBase):
    def __init__(self,
                 num_stages,
                 in_channels,
                 inner_channels,
                 norm_cfg=None,
                 conv_cfg=None,
                 up2x_num=0,
                 **kwargs):
        super(PanoFpnHead, self).__init__(
            **kwargs
        )
        self.up2x_num = up2x_num
        self.subnet = nn.ModuleList()
        self.subnet.append(
            ConvModule(in_channels, inner_channels, 3, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        )
        for i in range(1, num_stages):
            self.subnet.append(
                FCNSubNet(
                    in_channels,
                    inner_channels,
                    num_layers=i,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )
        self.upsample_layer = None
        if up2x_num>0:
            self.upsample_layer =  FCNSubNet(
                    in_channels,
                    inner_channels,
                    num_layers=self.up2x_num,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )

#         self.score = nn.Conv2d(inner_channels, self.num_classes, 1)
        self.init_weights()

#     def init_weights(self):
#         nn.init.normal_(self.score.weight.data, 0, 0.01)
#         self.score.bias.data.zero_()

    @force_fp32(apply_to=('features',))
    def forward(self, x):
        features = []
        for i, f in enumerate(x):
            features.append(self.subnet[i](f))
        features = torch.sum(torch.stack(features, dim=0), dim=0)
        if self.upsample_layer:
            features = self.upsample_layer(features)

#         score = self.score(features)
#         ret = {'fcn_score': score, 'fcn_feat': features}
        return features
