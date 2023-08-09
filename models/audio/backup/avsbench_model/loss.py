import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def bce_loss_with_ignore(pred_masks, first_gt_mask, T=5):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    # assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]
    # indices = torch.tensor(list(range(0, len(pred_masks), T)))

    # first_pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    # assert first_pred.requires_grad == True, "Error when indexing predited masks"
    # if len(first_gt_mask.shape) == 5:
    #     first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]
    weight_pos = (first_gt_mask != 255).long()
    # weight_neg = (first_gt_mask == 255).long() * 0.2
    first_bce_loss = nn.BCELoss(weight=weight_pos)(
        pred_masks, first_gt_mask.float()
    )
    return first_bce_loss


def A_MaskedV_SimmLoss(
    pred_masks,
    a_fea_list,
    v_map_list,
    count_stages=[],
    mask_pooling_type="avg",
    norm_fea=True,
):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_AV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [B*5, 1, 224, 224]
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage]
        a_fea = a_fea.view(-1, a_fea.shape[-1])  # [B*5, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert (
            C == a_fea.shape[-1]
        ), "Error: dimensions of audio and visual features are not equal"

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(
                pred_masks
            )  # [bs*5, 1, H, W]
        elif mask_pooling_type == "max":
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(
                pred_masks
            )  # [bs*5, 1, H, W]
        downsample_pred_masks = (downsample_pred_masks > 0.5).float()  # [bs*5, 1, H, W]

        obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1)  # [bs*5, 1]

        masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*5, C, H, W]
        # masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*5, C]
        masked_v_fea = masked_v_map.sum(-1).sum(-1) / (
            obj_pixel_num + 1e-6
        )  # [bs*5, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        cos_simm_va = torch.sum(torch.mul(masked_v_fea, a_fea), dim=-1)  # [bs*5]
        cos_simm_va = F.relu(cos_simm_va) + 1e-6
        cos_simm_va = (-1) * cos_simm_va.log()
        loss = cos_simm_va.mean()
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss


def SegLoss(
    pred_masks,
    first_gt_mask,
):
    """
    loss for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    total_loss = 0
    bce_loss = bce_loss_with_ignore(pred_masks, first_gt_mask)
    # f1_iou_loss = nn.BCELoss()(pred_masks, first_gt_mask)

    total_loss += bce_loss

    loss_dict = {}

    return total_loss, loss_dict


if __name__ == "__main__":

    pdb.set_trace()
