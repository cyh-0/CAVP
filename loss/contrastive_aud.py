from abc import ABC

import torch
import torch.nn as nn


class ContrastLoss(nn.Module, ABC):
    def __init__(self, temperature, ignore_idx, max_views):
        super(ContrastLoss, self).__init__()
        self.ignore_idx = ignore_idx
        self.ood_idx = 254
        self.eps = 1e-12
        """ Ablation """
        self.temperature = temperature
        self.max_views = max_views

    def forward(self, embeds_match, gt_match, embeds_shuffle, gt_shuffle):
        gt_match = torch.nn.functional.interpolate(gt_match.unsqueeze(1).float(), size=embeds_match.shape[2:],
                                                   mode='nearest').squeeze().long()

        gt_shuffle = torch.nn.functional.interpolate(gt_shuffle.unsqueeze(1).float(), size=embeds_shuffle.shape[2:],
                                                     mode='nearest').squeeze().long()

        # normalise the embed results
        embeds_match = torch.nn.functional.normalize(embeds_match, p=2, dim=1)
        embeds_shuffle = torch.nn.functional.normalize(embeds_shuffle, p=2, dim=1)
        
        # # randomly extract embed samples within a batch
        anchor_embeds, anchor_labels, contrs_embeds, contrs_labels = self.extraction_samples(embeds_match, gt_match,
                                                                                             embeds_shuffle, gt_shuffle)

        # calculate the CoroCL
        loss = self.info_nce(anchors_=anchor_embeds, a_labels_=anchor_labels.unsqueeze(1), contras_=contrs_embeds,
                             c_labels_=contrs_labels.unsqueeze(1)) if anchor_embeds.nelement() > 0 else \
            torch.tensor([.0], device=gt_match.device)

        return loss

    # The implementation of cross-image contrastive learning is based on:
    # https://github.com/tfzhou/ContrastiveSeg/blob/287e5d3069ce6d7a1517ddf98e004c00f23f8f99/lib/loss/loss_contrast.py
    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        # calculates the binary mask: same category => 1, different categories => 0
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()
        # calculates the dot product
        anchor_dot_contrast = torch.div(torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
                                        self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # calculates the negative mask
        neg_mask = 1 - mask

        # avoid the self duplicate issue
        mask = mask.fill_diagonal_(0.)

        # sum the negative odot results
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        
        # logits -> log(exp(x*x.T))
        # log_prob -> log(exp(x))-log(exp(x) + exp(y))
        # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(exp_logits + neg_logits)
        assert ~torch.isnan(log_prob).any(), "nan check 1."

        # calculate the info-nce based on the positive samples (under same categories)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+self.eps)
        assert ~torch.isnan(mean_log_prob_pos).any(), "nan check 2."

        return - mean_log_prob_pos.mean()

    def foreground_random_selection(self, embed_match_foregrounds_, gt_match_foregrounds_):
        categories = torch.unique(gt_match_foregrounds_)
        per_class_sample_num = self.max_views  # int(self.max_views / categories.shape[0])
        chosen_samples = []
        chosen_labels = []
        for item in categories:
            current_idx = gt_match_foregrounds_ == item
            if torch.sum(current_idx) < self.max_views: continue
            current_labels = gt_match_foregrounds_[current_idx]
            current_samples = embed_match_foregrounds_[current_idx]
            rand_idx = torch.randperm(current_labels.shape[0])
            chosen_labels.append(current_labels[rand_idx][:per_class_sample_num])
            chosen_samples.append(current_samples[rand_idx][:per_class_sample_num])
        return chosen_samples, chosen_labels

    def extraction_samples(self, embeds_match_, gt_match_, embeds_shuffle_, gt_shuffle_):
        # reformat the matrix
        embeds_match_ = embeds_match_.flatten(start_dim=2).permute(0, 2, 1)
        gt_match_ = gt_match_.flatten(start_dim=1)
        embeds_shuffle_ = embeds_shuffle_.flatten(start_dim=2).permute(0, 2, 1)
        gt_shuffle_ = gt_shuffle_.flatten(start_dim=1)

        embed_match_foregrounds = embeds_match_[(gt_match_ > 0) & (gt_match_ != self.ignore_idx)]
        gt_match_foregrounds = gt_match_[(gt_match_ > 0) & (gt_match_ != self.ignore_idx)]

        chosen_embed_match_foregrounds, chosen_gt_match_foregrounds = \
            self.foreground_random_selection(embed_match_foregrounds, gt_match_foregrounds)

        if len(chosen_embed_match_foregrounds) > 0 and len(chosen_gt_match_foregrounds) > 0:
            chosen_embed_match_foregrounds = torch.cat(chosen_embed_match_foregrounds)
            chosen_gt_match_foregrounds = torch.cat(chosen_gt_match_foregrounds)
        else:
            return torch.empty(0), None, None, None

        embed_match_backgrounds = embeds_match_[gt_match_ == 0]
        gt_match_backgrounds = gt_match_[gt_match_ == 0]

        # based on the original mask to select the potential silent object (which shouldn't be segmented).
        # gt shuffle should be 0 mask when the sound doesn't match.
        embed_shuffle_ = embeds_shuffle_[(gt_match_ > 0) & (gt_match_ != self.ignore_idx)]
        gt_shuffle_ = gt_shuffle_[(gt_match_ > 0) & (gt_match_ != self.ignore_idx)]

        # define the number of choice.
        sample_num = int(min(self.max_views, embed_shuffle_.shape[0], embed_match_backgrounds.shape[0]))

        # select the index randomly.
        shuffle_idx_1 = torch.randperm(embed_match_backgrounds.shape[0])
        shuffle_idx_2 = torch.randperm(embed_shuffle_.shape[0])

        # anchors = torch.cat([chosen_embed_match_foregrounds,
        #                      embed_match_backgrounds[shuffle_idx_1][:sample_num]], dim=0)

        # labels = torch.cat([chosen_gt_match_foregrounds,
        #                     gt_match_backgrounds[shuffle_idx_1][:sample_num]], dim=0)

        # build the anchors and labels based on the selected indices.
        """ Avoid paired feature to cluster with other background """
        anchors = torch.cat([chosen_embed_match_foregrounds,
                             embed_match_backgrounds[shuffle_idx_1][:sample_num],
                             embed_shuffle_[shuffle_idx_2][:sample_num]
                             ], dim=0)
        
        labels = torch.cat([chosen_gt_match_foregrounds,
                            gt_match_backgrounds[shuffle_idx_1][:sample_num],
                            gt_shuffle_[shuffle_idx_2][:sample_num]], dim=0)

        return anchors, labels, anchors.clone(), labels.clone()