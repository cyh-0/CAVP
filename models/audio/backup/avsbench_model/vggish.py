import torch

from models.audio.backup.avsbench_model.config import cfg
from models.audio.backup.avsbench_model.torchvggish import vggish


class audio_extractor(torch.nn.Module):
    def __init__(self, args):
        super(audio_extractor, self).__init__()
        self.feature_extactor = vggish.VGGish(cfg, args.local_rank)
        self.fc = torch.nn.Linear(128, args.num_classes)

    def forward(self, audio):
        audio_fea = self.feature_extactor(audio)
        return self.fc(audio_fea)

