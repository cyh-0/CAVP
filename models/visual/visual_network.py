import torch.nn as nn
from models.visual.deeplabv3.encoder_decoder import Backbone, DeepLabV3Plus
from models.visual.backbones.hrnet.hrnet import hrnet_w48
from models.visual.hrnet.hrnetv2_w48 import HRNet_W48
from models.visual.ocrnet.ocrnet import OCR
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
import torch

"""
[50, 101]: [DeepLabV3Plus]
hrnet_w48: [HRNet_W48, OCR] 
"""


class VisualModel(nn.Module):
    def __init__(
        self,
        backbone,
        pretrain_path,
        num_classes=2,
        ignore_index=255,
        seg_model="DeepLabV3Plus",
        last_three_dilation_stride=[False, True, True],
    ):
        super(VisualModel, self).__init__()
        logger.critical(f"LOADING SEG MODEL <<{seg_model}>>")
        if seg_model == "DeepLabV3Plus":
            self.backbone = Backbone(
                back_bone=backbone,
                norm_layer=nn.BatchNorm2d,
                pretrained_model=pretrain_path,
                last_three_dilation_stride=last_three_dilation_stride,
            )
            self.segment = DeepLabV3Plus(
                num_classes=num_classes,
                aspp_in_plane=2048 if backbone == 50 or backbone == 101 else 512,
                aspp_out_plane=256 if backbone == 50 or backbone == 101 else 64,
            )
        elif seg_model == "HRNet":
            self.backbone = hrnet_w48(pretrain_path)
            self.segment = HRNet_W48(num_classes=num_classes)
        elif seg_model == "OCR":
            self.backbone = hrnet_w48(pretrain_path)
            self.segment = OCR(num_classes=num_classes)
        else:
            raise ValueError("UNKNOW BACKBONE")

        self.ignore_index = ignore_index

    def forward(self, image, audio=None):
        input_shape = image.shape[-2:]
        x_fea = self.backbone(image)
        # a = torch.nn.functional.normalize(audio, dim=1, p=2)
        # i = torch.nn.functional.normalize(x_fea[-1], dim=1, p=2)
        # proj_ = torch.einsum('ncqa,nchw->nqa', [i, a.unsqueeze(2).unsqueeze(3)]).unsqueeze(1)
        # x_fea[-1] = x_fea[-1] + x_fea[-1] * proj_
        out = self.segment(x_fea)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out
