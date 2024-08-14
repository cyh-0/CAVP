import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
from torchvision import transforms

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import pdb


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float()  # [BF, H, W]
    intersection = predict * (predict == target).float()  # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(
            intersection[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP
        area_pred = torch.histc(
            predict[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP + FP
        area_lab = torch.histc(
            target[i].cpu(), bins=nbins, min=mini, max=maxi)  # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item(
        ) == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / \
            (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum(iou != 0).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1)  # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T)
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(
        pred, target, nclass, T)
    return miou, fscore, cls_count, vid_miou_list


def _batch_intersection_union(output, target, nclass, T):
    """mIoU"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    # pdb.set_trace()

    predict = predict.float() * (target > 0).float()  # [BF, H, W]
    intersection = predict * (predict == target).float()  # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    for i in range(target.shape[0]):
        area_inter = torch.histc(
            intersection[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(
            predict[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item(
        ) == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1
        # pdb.set_trace()
    # ious = ious / cls_count
    # ious[torch.isnan(ious)] = 0
    # pdb.set_trace()
    # return area_inter.float(), area_union.float()
    # return ious
    return ious, cls_count


def calc_color_miou(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1)  # [BF, C, H, W]
    # correct, labeled = _batch_pix_accuracy(pred, target)
    # inter, union = _batch_intersection_union(pred, target, nclass, T)
    ious, cls_count = _batch_intersection_union(pred, target, nclass, T)

    # pixAcc = 1.0 * correct / (2.220446049250313e-16 + labeled)
    # IoU = 1.0 * inter / (2.220446049250313e-16 + union)
    # mIoU = IoU.mean().item()
    # pdb.set_trace()
    # return mIoU
    return ious, cls_count


def calc_binary_miou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x C x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # assert len(pred.shape) == 3 and pred.shape == target.shape
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1)  # [BF, C, H, W]
    pred = torch.argmax(pred, dim=1)  # [BF, H, W]
    binary_pred = (pred != (nclass - 1)).float()  # [BF, H, W]
    # pdb.set_trace()
    pred = binary_pred
    target = (target != (nclass - 1)).float()

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N
    # pdb.set_trace()
    return iou


import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
import pdb
from torchvision import transforms

logger = logging.getLogger(__name__)


def save_checkpoint(state, epoch, is_best, checkpoint_dir='./models', filename='checkpoint', thres=100):
    """
    - state
    - epoch
    - is_best
    - checkpoint_dir: default, ./models
    - filename: default, checkpoint
    - freq: default, 10
    - thres: default, 100
    """
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if epoch >= thres:
        file_path = os.path.join(
            checkpoint_dir, filename + '_{}'.format(str(epoch)) + '.pth.tar')
    else:
        file_path = os.path.join(checkpoint_dir, filename + '.pth.tar')
    torch.save(state, file_path)
    logger.info('==> save model at {}'.format(file_path))

    if is_best:
        cpy_file = os.path.join(
            checkpoint_dir, filename + '_model_best.pth.tar')
        shutil.copyfile(file_path, cpy_file)
        logger.info('==> save best model at {}'.format(cpy_file))


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    # temp_pred = torch.sigmoid(pred)
    # pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1) # bg inter
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, cuda_flag=True):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / \
            (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path="", pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    assert len(pred.shape) == 3 and pred.shape == gt.shape
    # pred = torch.sigmoid(pred) # =======================================[important]
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    # fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0 # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    # fLog.close()

    return score.max().item()


def save_mask(pred_masks, save_base_path, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()

    pred_masks = pred_masks.view(-1, 5,
                                 pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx]  # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png" % (video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')


def save_raw_img_mask(anno_file_path, raw_img_base_path, mask_base_path, split='test', r=0.5):
    df = pd.read_csv(anno_file_path, sep=',')
    df_test = df[df['split'] == split]
    count = 0
    for video_id in range(len(df_test)):
        video_name = df_test.iloc[video_id][0]
        raw_img_path = os.path.join(raw_img_base_path, video_name)
        for img_id in range(5):
            img_name = "%s.mp4_%d.png" % (video_name, img_id + 1)
            raw_img = cv2.imread(os.path.join(raw_img_path, img_name))
            mask = cv2.imread(os.path.join(
                mask_base_path, 'pred_masks', video_name, "%s_%d.png" % (video_name, img_id)))
            # pdb.set_trace()
            raw_img_mask = cv2.addWeighted(raw_img, 1, mask, r, 0)
            save_img_path = os.path.join(
                mask_base_path, 'img_add_masks', video_name)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_img_path, img_name), raw_img_mask)
        count += 1
    print(f'count: {count} videos')
