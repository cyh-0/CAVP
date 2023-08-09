import torch
import torch.nn as nn
import torch.nn.functional as F
from models.visual.backbones.hrnet.hrnet import HighResolutionNet

# self.upsample = nn.Sequential(
#     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(in_channels),
#     nn.ReLU(inplace=True),
#     nn.Dropout2d(0.10),
# )
#     nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)

class Upsampling(nn.Module):
    def __init__(self, classifier_in_channels, num_classes, conv_in, norm_act=nn.BatchNorm2d):
        super(Upsampling, self).__init__()
        self.classifier = nn.Conv2d(classifier_in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.last_conv = nn.Sequential(
            nn.Conv2d(conv_in, conv_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_in),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
        )

    def forward(self, x):
        f = self.last_conv(x).contiguous()
        return self.classifier(f)


class HRNet_W48(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, num_classes):
        super(HRNet_W48, self).__init__()
        self.num_classes = num_classes
        # self.backbone = HighResolutionNet()

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.upsample = Upsampling(in_channels, num_classes, conv_in=720, norm_act=torch.nn.BatchNorm2d)
        
        self.business_layer = []
        self.business_layer.append(self.upsample)

    def forward_feature(self, x):
        # x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)

        return feats

    def forward(self, x):
        feats = self.forward_feature(x)
        # out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return self.upsample(feats)

