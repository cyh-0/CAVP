from loss.contrastive import PixelContrastLoss
from loss.av_contrast import AVContrast
from loss.sup_contrastive import SupConLoss
import torch.nn.functional as F
from torch import nn
import torch

class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class Losser(nn.Module):
    def __init__(self, num_classes, local_rank=0):
        super(Losser, self).__init__()
        self.num_classes = num_classes
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=255)
        self.loss_supcl = SupConLoss(local_rank)
        self.loss_ce_audio = nn.MultiLabelSoftMarginLoss()
        self.loss_contrastive = PixelContrastLoss(0.1, local_rank)
        self.loss_av_contrast = AVContrast(0.1, local_rank)
        self.local_rank = local_rank

    def forward(self, output, pix_label):
        l_ce = self.loss_ce(output, pix_label)

        # multi_label_y = [F.one_hot(item, self.num_classes).sum(0) for item in target]
        # multi_label_y = torch.stack(multi_label_y).to(output.device)
        # multi_label_y[:, 0] = 0
        # loss_aud_cls = self.loss_ce_audio(aud_pred, multi_label_y)
        loss_aud_cls = torch.tensor([0.0], device=self.local_rank)

        # target = [item[item != 0] for item in target]
        # target = torch.stack(target).view(-1)

        # Global contrastive learning between feature
        # norm_f = F.normalize(f_a_a2v, dim=-1)
        # l_contrast_global = self.loss_supcl(norm_f, target)
        l_contrast_global = torch.tensor([0.0], device=self.local_rank)

        # Pixel contrastive learning
        # l_contrast_pixel = self.loss_contrastive(v_proj, predict=output, labels=pix_label)
        l_contrast_pixel = torch.tensor([0.0], device=self.local_rank)

        # Mask contrastive learning
        # l_contrast_av = self.loss_av_contrast(v_proj, a_proj, labels=pix_label)
        l_contrast_av = torch.tensor([0.0], device=self.local_rank)

        return l_ce, l_contrast_global, l_contrast_pixel, loss_aud_cls, l_contrast_av
