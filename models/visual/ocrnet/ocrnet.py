##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.visual.ocrnet.ocr_module import SpatialGather_Module, SpatialOCR_Module


class OCR(nn.Module):
    def __init__(self, num_classes):
        super(OCR, self).__init__()
        self.num_classes = num_classes
        # self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 720
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=nn.BatchNorm2d)

        self.upsample = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.business_layer = []
        self.business_layer.append(self.conv3x3)
        self.business_layer.append(self.ocr_gather_head)
        self.business_layer.append(self.ocr_distri_head)
        self.business_layer.append(self.upsample)
        self.business_layer.append(self.aux_head)

    def forward_feature(self, x):
        # x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        return feats

    def forward(self, x):
        feats = self.forward_feature(x)
        # out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        # out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        # return out_aux, out
        return self.upsample(feats)

