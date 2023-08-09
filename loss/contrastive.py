import torch
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, temp1, local_rank):
        super(PixelContrastLoss, self).__init__()
        self.temperature = temp1
        # make it to be 1; we don't want to fine-tune it.
        self.base_temperature = self.temperature
        self.ignore_label = 255
        self.max_samples = 1024
        self.max_views = 100
        self.local_rank = local_rank

    def _hard_anchor_sampling(self, X, y, y_hat, conf):
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0

        # go through batch
        for ii in range(batch_size):
            # for 1 image for the unique
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            # filter out ignore
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x
                for x in this_classes
                if (this_y == x).nonzero().shape[0] > self.max_views
            ]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # n_view = self.max_samples // total_classes
        # n_view = min(n_view, self.max_views)
        n_view = self.max_views

        # total_classes -> the classes within the batch
        # n_view -> the sample within those classes
        # the pre-defined feature
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float, device=self.local_rank)
        y_ = torch.zeros(total_classes, dtype=torch.float, device=self.local_rank)
        c_ = torch.zeros((total_classes, n_view, 1), dtype=torch.float, device=self.local_rank)
        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            # Mine HARD and EASY contrastive postive pairs
            for cls_id in this_classes:
                # false positive
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                # true positive
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                c_[X_ptr] = conf[ii, indices]
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_, c_

    def _contrastive(self, feats_, labels_, conf_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)
        # same class be positive; different classes be negative
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda(self.local_rank)
        contrast_count = n_view
        # 4x100x256 -> 400x256
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # positive belongs to itself
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature,
        )
        # for numerical stability, but why?
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        # avoid the self duplicate issue
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(self.local_rank), 0
        )
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)  # * mask
        # define logits => x
        # x [400x400] <- all_pairs and x=log(exp(x))
        # y [400x1] <- sum of all negative pairs
        # log_prob -> log(exp(x))-log(exp(x) + exp(y))
        # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        # log_prob [400x400] -> each sample is a pair

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, predict=None, labels=None):
        """
        feats -> projected visual feature (to audio space)
        """
        feats = F.normalize(feats, p=2, dim=1)
        predict = torch.nn.functional.interpolate(
            predict, (feats.shape[2], feats.shape[3]), mode="bilinear"
        )
        conf, predict = torch.max(torch.softmax(predict, dim=1), dim=1)

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(
            labels, (feats.shape[2], feats.shape[3]), mode="nearest"
        )
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], "{} {}".format(
            labels.shape, feats.shape
        )

        batch_size = feats.shape[0]

        conf = conf.contiguous().view(batch_size, -1)
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats_, labels_, conf_ = self._hard_anchor_sampling(
            feats, predict, labels, conf
        )
        loss = self._contrastive(feats_, labels_, conf_)
        return loss
