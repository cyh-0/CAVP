from models.audio.backup.resnet import resnet18 as resnet18_mod
import torch
import torch.nn as nn


class audio_model(nn.Module):
    def __init__(self, args, num_classes=20):
        super(audio_model, self).__init__()
        self.args = args
        self.backbone = resnet18_mod(modal='audio')
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def get_feature(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.get_feature(x)
        x = self.fc(x)
        return x



