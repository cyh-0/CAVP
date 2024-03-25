import torch.nn as nn
from models.visual.deeplabv3.encoder_decoder import Backbone, DeepLabV3Plus
from models.visual.backbones.hrnet.hrnet import hrnet_w48
from models.visual.hrnet.hrnetv2_w48 import HRNet_W48
from models.visual.ocrnet.ocrnet import OCR
from einops import rearrange
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# from config import config
from models.visual.backbones.resnet import resnet50, resnet18
from models.audio.audio_network import AudioModel
# from models.audio.audio_network_vggish import AudioModel
from models.attn import CROSS_ATTENTION, Mlp
from loguru import logger


class SoundBank:
    def __init__(self, out_dim=304, args=None, device=0):
        self.sound_bank = torch.zeros(
            (args.num_classes, args.batch_size, out_dim),
            requires_grad=False,
            device=device,
        )

    def update_bank(self, waveform, img_label):
        img_label[:, 0] = 0
        target = [
            item.nonzero()
            .squeeze()
            .cpu()
            .view(
                -1,
            )
            .tolist()
            for item in img_label
        ]
        for i in range(len(target)):
            item = target[i]
            for sub_label in item:
                self.queue(sub_label, waveform[i, None])
        return

    def queue(self, class_idx, fea_a):
        self.sound_bank[class_idx] = torch.cat(
            (self.sound_bank[class_idx][1:], fea_a.detach()), dim=0
        )
        return

    def overwrite_audio_feature(self, shuffle_fea_a, org_fea_a, mod_idx_map):
        for i, (idx, target_label) in enumerate(mod_idx_map.items()):
            """Change audio at <idx> to random audio with label <curr_label>"""
            fake_audio = self.sound_bank[None, target_label][:, 0]
            # self.queue(target_label, org_fea_a[idx, None])
            shuffle_fea_a[idx] = fake_audio
        return shuffle_fea_a


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, norm_act=nn.BatchNorm2d):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            norm_act(dim_in),
            # nn.ReLU(),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.proj(x)


class CAVP(nn.Module):
    def __init__(
        self,
        backbone,
        pretrain_path,
        num_classes=2,
        ignore_index=255,
        audio_backbone_pretrain_path=None,
        visual_backbone=50,
        args=None,
    ):
        super(CAVP, self).__init__()
        seg_model = args.seg_model
        last_three_dilation_stride = args.last_three_dilation_stride
        logger.critical(f"LOADING SEG MODEL <<{seg_model}>>")
        if seg_model == "DeepLabV3Plus":
            self.latent_dim = 304
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
            self.latent_dim = 720
            self.backbone = hrnet_w48(pretrain_path)
            self.segment = HRNet_W48(num_classes=num_classes)
        elif seg_model == "OCR":
            self.latent_dim = 512
            self.backbone = hrnet_w48(pretrain_path)
            self.segment = OCR(num_classes=num_classes)
        else:
            raise ValueError("UNKNOW BACKBONE")

        self.cross_att = CROSS_ATTENTION(
            dim_in=self.latent_dim, embed_dim=self.latent_dim, depth=1
        )
        # self.audio_projector = Mlp(
        #     in_features=128,
        #     hidden_features=304,
        #     out_features=304,
        #     drop=0.0,
        # )
        self.visual_projector = Mlp(
            in_features=self.latent_dim,
            hidden_features=256,
            out_features=self.latent_dim,
            drop=0.0,
        )
        # self.visual_projector = ProjectionHead(self.latent_dim, self.latent_dim)

        self.audio_backbone = AudioModel(
            args.audio_backbone, audio_backbone_pretrain_path, self.latent_dim
        )
        self.memory = SoundBank(
            out_dim=self.latent_dim, args=args, device=args.local_rank
        )

    def forward_cls(self, out, input_shape):
        out = self.segment.upsample(out.contiguous())
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out

    def forward_fusion(self, visual, fea_a):
        b, c, h, w = visual.shape
        visual = rearrange(visual, "b c h w -> b (h w) c", h=h, w=w)
        fea_v = self.visual_projector(visual)
        fea_v = rearrange(fea_v, "b (h w) c -> b c h w", h=h, w=w)
        # fea_v_proj = fea_v.clone()

        # audio_feature = fea_a.clone()
        fea_a = fea_a.unsqueeze(-1).unsqueeze(-1)
        fea_v, fea_a, attn_v = self.cross_att(fea_v, fea_a)
        fea_v = rearrange(fea_v, "b (h w) c -> b c h w", h=h, w=w)
        # fea_a = rearrange(fea_a, "b (h w) c -> b c h w", h=1, w=1)
        """
        remove
        """
        # v_proj = rearrange(fea_v_proj, "b c h w -> b (h w) c", h=h, w=w)
        # a_proj = self.audio_proj(audio_feature)
        # a_proj = audio_feature
        return fea_v, attn_v

    def forward_audio(self, audio, shuffle_info=None, ow_flag=False):
        fea_a = self.audio_backbone(audio)
        # fea_a = self.audio_projector(fea_a)
        # """ Shuffled audio + replace some audio with bank item """
        shuffle_idx = shuffle_info["shuffle_idx"]
        mod_idx_map = shuffle_info["mod_idx_map"]
        image_label = shuffle_info["image_label"]
        shuffle_fea_a = fea_a.clone().detach()[shuffle_idx]
        """ Update memory bank """
        if ow_flag:
            shuffle_fea_a = self.memory.overwrite_audio_feature(
                shuffle_fea_a, fea_a, mod_idx_map
            )
            # self.memory.update_bank(fea_a, mod_idx_map)
            self.memory.update_bank(fea_a, image_label)
        shuffle_fea_a = fea_a[shuffle_idx]

        return torch.cat((fea_a, shuffle_fea_a), dim=0)

    def forward_train(self, image, audio=None, shuffle_info=None, ow_flag=False):
        B = image.shape[0]
        input_shape = image.shape[-2:]
        x_fea = self.backbone(image)
        fea_v = self.segment.forward_feature(x_fea)
        """Concatenate features perpared for contrstive learning"""
        fea_v = torch.cat((fea_v, fea_v.clone()), dim=0)
        fea_a = self.forward_audio(audio, shuffle_info, ow_flag)
        out_fusion, attn_v = self.forward_fusion(fea_v, fea_a)
        out_pred = self.forward_cls(out_fusion, input_shape)
        return out_pred, out_fusion, attn_v

    def forward_inference(self, image, audio=None):
        input_shape = image.shape[-2:]
        x_fea = self.backbone(image)
        fea_v = self.segment.forward_feature(x_fea)
        fea_a = self.audio_backbone(audio)
        out_fusion, _ = self.forward_fusion(fea_v, fea_a)
        out_pred = self.forward_cls(out_fusion, input_shape)
        return out_pred, out_fusion

    def forward(
        self, image, audio=None, shuffle_info=None, ow_flag=False, eval_mode=False
    ):
        if eval_mode:
            return self.forward_inference(image, audio)
        else:
            return self.forward_train(image, audio, shuffle_info, ow_flag)
