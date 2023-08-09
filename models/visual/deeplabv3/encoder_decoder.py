from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
# from config import config
from models.visual.backbones.resnet import resnet18, resnet50, resnet101

bn_eps = 1e-5
bn_momentum = 0.1


class Backbone(nn.Module):
    def __init__(self, back_bone, norm_layer=nn.BatchNorm2d, pretrained_model=None,
                 last_three_dilation_stride=None):
        super(Backbone, self).__init__()
        logger.critical(f"Last three dilation stride set to {last_three_dilation_stride}")
        if back_bone == 18:
            self.backbone = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=True, stem_width=64,
                                     replace_stride_with_dilation=last_three_dilation_stride)
        elif back_bone == 50:
            self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=True, stem_width=64,
                                     replace_stride_with_dilation=last_three_dilation_stride)
        elif back_bone == 101:
            self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=True, stem_width=64,
                                     replace_stride_with_dilation=last_three_dilation_stride)
        else:
            raise ValueError
        
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, data):
        f_list = self.backbone(data)
        return f_list


class Upsampling(nn.Module):
    def __init__(self, classifier_in_channels, num_classes, conv_in, norm_act=nn.BatchNorm2d):
        super(Upsampling, self).__init__()
        self.classifier = nn.Conv2d(classifier_in_channels, num_classes, kernel_size=1, bias=True)
        self.last_conv = nn.Sequential(nn.Conv2d(conv_in, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU())

    def forward(self, x):
        f = self.last_conv(x).contiguous()
        return self.classifier(f)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, aspp_in_plane=2048, aspp_out_plane=256, classifier_in=256, norm_layer=nn.BatchNorm2d):
        super(DeepLabV3Plus, self).__init__()
        conv_in = 112 if aspp_out_plane == 64 else 304
        self.aspp = ASPP(aspp_in_plane, aspp_out_plane, [6, 12, 18], norm_act=norm_layer)

        self.reduce = nn.Sequential(
            nn.Conv2d(aspp_out_plane, 48, 1, bias=False),
            norm_layer(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.upsample = Upsampling(classifier_in, num_classes, conv_in=conv_in, norm_act=torch.nn.BatchNorm2d)

        self.business_layer = []
        self.business_layer.append(self.aspp)
        self.business_layer.append(self.reduce)
        self.business_layer.append(self.upsample.last_conv)
        self.business_layer.append(self.upsample.classifier)

    def forward_feature(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)
        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)
        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        return f

    def forward(self, f_list):
        f = self.forward_feature(f_list)
        return self.upsample(f)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(12, 24, 36), hidden_channels=256,
                 norm_act=nn.BatchNorm2d, pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size
        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)  # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.leak_relu(pool)  # add activation layer
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            raise NotImplementedError
        return pool
