import torch
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AVContrast(nn.Module, ABC):
    def __init__(self, temp1, local_rank):
        super(AVContrast, self).__init__()
        self.temperature = temp1
        # make it to be 1; we don't want to fine-tune it.
        self.base_temperature = self.temperature
        self.ignore_label = 255
        self.max_samples = 1024
        self.max_views = 100
        self.eps = 1e-12
        self.local_rank = local_rank

    def _contrastive(self, audio, visual, label):
        device = audio.device
        # [B, 2, C]
        features = torch.cat((audio.unsqueeze(1), visual.unsqueeze(1)), dim=1)

        batch_target = [torch.unique(item) for item in label]
        batch_target = [item[item != self.ignore_label] for item in batch_target]
        batch_target = [item[item != 0] for item in batch_target]

        # features_list = []
        # target_list = []
        zero_idx = []
        for i in range(len(batch_target)):
            if len(batch_target[i]) == 0:
                zero_idx.append(i)
                batch_target[i] = torch.tensor([255], device=device)

            # if len(batch_target[i]) != 0:
            #     features_list.append(features[i])
            #     target_list.append(batch_target[i])
            # else:
            #     a=1

        # features = torch.stack(features_list)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        batch_size = features.shape[0]

        batch_target = torch.stack(batch_target).view(-1, 1)
        mask = torch.eq(batch_target, batch_target.T).float().cuda(self.local_rank)

        for i in zero_idx:
            mask[i]=0

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # mask average pooling
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def forward(self, f_v, f_a, labels=None):
        """
        feats -> projected visual feature (to audio space)
        """
        h, w = 128, 128
        b, hw, c = f_v.shape
        batch_size = b
        f_v = F.normalize(f_v, p=2, dim=1)
        f_a = F.normalize(f_a, p=2, dim=1)
        # predict = torch.nn.functional.interpolate(predict, (f_v.shape[2], f_v.shape[3]), mode='bilinear')
        # conf, predict = torch.max(torch.softmax(predict, dim=1), dim=1)

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (h, w), mode="nearest")
        labels = labels.squeeze(1).long()
        # assert labels.shape[-1] == f_v.shape[-1], '{} {}'.format(labels.shape, f_v.shape)

        labels = rearrange(labels, "b h w -> b (h w)", h=h, w=w)

        mask = torch.ones_like(labels).cuda(self.local_rank) - (
            (labels == 0).long() + (labels == self.ignore_label).long()
        )

        masked_v = torch.mul(mask.unsqueeze(-1), f_v)
        masked_v = torch.div(masked_v.sum(1), (mask.sum(1).unsqueeze(-1) + self.eps))

        loss = self._contrastive(f_a, masked_v, labels)
        return loss
