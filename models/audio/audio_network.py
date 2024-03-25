import torch
import torch.nn as nn
from models.audio.backbones.vgg import VGG
from models.attn import Mlp
from torchvision.models import resnet18
from models.visual.deeplabv3.encoder_decoder import Backbone
from loguru import logger

class AudioModel(torch.nn.Module):
    def __init__(self, backbone, pretrain_path, out_plane, num_classes=2, in_plane=1):
        super(AudioModel, self).__init__()
        if backbone == "vgg":
            self.backbone = VGG(out_plane)
            if pretrain_path is not None:
                logger.warning(f'==> Load pretrained VGG parameters from {pretrain_path}')
                # self.backbone.load_state_dict(torch.load(pretrain_path), strict=True)
                self.load_audio_model(path_=pretrain_path)
        else:
            logger.warning(f'==> Load pretrained ResNet18 for Audio')
            self.backbone = resnet18(True)
            self.backbone.conv1 = nn.Conv2d(
                in_plane, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            self.backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.backbone.fc = nn.Linear(512, out_plane)


        self.cls_head = nn.Linear(out_plane, num_classes)

    def forward_cls(self, x):
        return self.cls_head(self.backbone(x))

    def forward(self, x):
        return self.backbone(x)

    def load_audio_model(self, path_):
        # return
        param_dict = torch.load(path_)
        param_ = self.backbone.state_dict()
        out_, in_ = param_["embeddings.4.weight"].shape
        param_dict["embeddings.4.weight"] = torch.nn.init.kaiming_normal_(
            torch.zeros(out_, in_)
        )
        param_dict["embeddings.4.bias"] = torch.zeros(out_)
        self.backbone.load_state_dict(param_dict, strict=True)
