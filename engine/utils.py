import os
import json

import torch
from torch.optim import *
import numpy as np
from sklearn import metrics
import math

import cv2
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from utils.pyt_utils import make_dir
import torchvision.transforms as transforms
from terminaltables import AsciiTable
from loguru import logger
import os
import torch.nn as nn
import numpy as np
import PIL
from utils.conv_2_5d import Conv2_5D_depth, Conv2_5D_disp


def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

    def __call__(self, tensor):
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.mul_(s).add_(m)
        return self.invTrans(tensor.clone())


class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []
        self.area = []

        self.ciou_small = []
        self.ciou_med = []
        self.ciou_large = []
        self.ciou_huge = []

        self.score = []

    def cal_CIOU(self, bboxes, score, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map * gtmap) / (
            np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))
        )

        self.cal_CIoU_area(bboxes, ciou)

        self.ciou.append(ciou)
        self.score.append(score)
        return (
            ciou,
            np.sum(infer_map * gtmap),
            (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))),
        )

    def cal_CIoU_area(self, bboxes, ciou):

        area = self.cal_area(bboxes)
        self.area.append(area)

        if area in range(0, 32**2):
            self.ciou_small.append(ciou)
        elif area in range(32**2, 96**2):
            self.ciou_med.append(ciou)
        elif area in range(96**2, 144**2):
            self.ciou_large.append(ciou)
        elif area in range(144**2, 10**10):
            self.ciou_huge.append(ciou)

    def finalize_AUC(self, ciou):
        cious = [np.sum(np.array(ciou) >= 0.05 * i) / len(ciou) for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self, ciou):
        ap50 = np.mean(np.array(ciou) >= 0.5)
        return ap50

    def finalize_precison_recall(self, ciou, confidence, confidence_thr):
        true_pos = 0
        false_pos = 0
        false_neg = 0

        for i in range(len(ciou)):
            if confidence[i] >= confidence_thr:
                if ciou[i] >= 0.5:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                false_neg += 1

        print(true_pos, false_pos, false_neg)

        precison = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        return precison, recall

    def finalize_cIoU(self, ciou):
        ciou = np.mean(np.array(ciou))
        return ciou

    def clear(self):
        self.ciou = []
        self.ciou_small = []
        self.ciou_med = []
        self.ciou_large = []
        self.ciou_huge = []

    def cal_area(self, bboxes):
        area_list = []
        for xmin, ymin, xmax, ymax in bboxes:
            area = (ymax - ymin) * (xmax - xmin)
            area_list.append(abs(area))
        return int(np.mean(area_list))


class EvaluatorFull(object):
    def __init__(
        self,
        iou_thrs=(0.5, 0.75),
        default_conf_thr=0.5,
        pred_size=0.5,
        pred_thr=0.5,
        results_dir="./results",
    ):
        super(EvaluatorFull, self).__init__()
        self.iou_thrs = iou_thrs
        self.default_conf_thr = default_conf_thr
        self.min_sizes = {
            "small": 0,
            "medium": 32**2,
            "large": 96**2,
            "huge": 144**2,
        }
        self.max_sizes = {
            "small": 32**2,
            "medium": 96**2,
            "large": 144**2,
            "huge": 10000**2,
        }

        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []

        self.results_dir = results_dir
        self.wandb_run = wandb

        # self.viz_save_dir = os.path.join("./results/visual", str(wandb.run.dir))
        # make_dir(self.viz_save_dir)

        # self.results_save_dir = (
        #     f"{results_dir}/results_conf"
        #     + str(default_conf_thr)
        #     + "_predsize"
        #     + str(pred_size)
        #     + "_predthr"
        #     + str(pred_thr)
        # )
        os.makedirs(results_dir, exist_ok=True)
        # os.makedirs(self.viz_save_dir, exist_ok=True)/
        # os.makedirs(self.results_save_dir, exist_ok=True)

    @staticmethod
    def calc_precision_recall(
        bb_list, ciou_list, confidence_list, confidence_thr, ciou_thr=0.5
    ):
        assert len(bb_list) == len(ciou_list) == len(confidence_list)
        true_pos, false_pos, false_neg = 0, 0, 0
        for bb, ciou, confidence in zip(bb_list, ciou_list, confidence_list):
            if bb == 0:
                # no sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected
                    false_pos += 1
            else:
                # sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected...
                    if ciou >= ciou_thr:  # ...in correct place
                        true_pos += 1
                    else:  # ...in wrong place
                        false_pos += 1
                else:
                    # no sounding objects detected
                    false_neg += 1

        precision = (
            1.0 if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
        )
        recall = 1.0 if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)

        return precision, recall

    def calc_ap(self, bb_list_full, ciou_list_full, confidence_list_full, iou_thr=0.5):

        assert len(bb_list_full) == len(ciou_list_full) == len(confidence_list_full)

        # for visible objects
        # ss = [i for i, bb in enumerate(bb_list_full) if bb > 0]
        # bb_list = [bb_list_full[i] for i in ss]
        # ciou_list = [ciou_list_full[i] for i in ss]
        # confidence_list = [confidence_list_full[i] for i in ss]

        precision, recall, skip_thr = [], [], max(1, len(ciou_list_full) // 200)
        for thr in np.sort(np.array(confidence_list_full))[:-1][::-skip_thr]:
            p, r = self.calc_precision_recall(
                bb_list_full, ciou_list_full, confidence_list_full, thr, iou_thr
            )
            precision.append(p)
            recall.append(r)
        precision_max = [np.max(precision[i:]) for i in range(len(precision))]
        ap = sum(
            [
                precision_max[i] * (recall[i + 1] - recall[i])
                for i in range(len(precision_max) - 1)
            ]
        )
        return ap

    def cal_auc(self, bb_list, ciou_list):
        ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        ciou = [ciou_list[i] for i in ss]
        cious = [np.sum(np.array(ciou) >= 0.05 * i) / len(ciou) for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def filter_subset(
        self, subset, name_list, area_list, bb_list, ciou_list, conf_list
    ):
        if subset == "visible":
            ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        elif subset == "non-visible/non-audible":
            ss = [i for i, bb in enumerate(bb_list) if bb == 0]
        elif subset == "all":
            ss = [i for i, bb in enumerate(bb_list) if bb >= 0]
        else:
            ss = [
                i
                for i, sz in enumerate(area_list)
                if self.min_sizes[subset] <= sz < self.max_sizes[subset]
                and bb_list[i] > 0
            ]

        if len(ss) == 0:
            return [], [], [], [], []

        name = [name_list[i] for i in ss]
        area = [area_list[i] for i in ss]
        bbox = [bb_list[i] for i in ss]
        ciou = [ciou_list[i] for i in ss]
        conf = [conf_list[i] for i in ss]

        return name, area, bbox, ciou, conf

    def finalize_stats(self):
        (
            name_full_list,
            area_full_list,
            bb_full_list,
            ciou_full_list,
            confidence_full_list,
        ) = self.gather_results()

        metrics = {}
        for iou_thr in self.iou_thrs:
            for subset in ["all", "visible", "small", "medium", "large", "huge"]:
                _, _, bb_list, ciou_list, conf_list = self.filter_subset(
                    subset,
                    name_full_list,
                    area_full_list,
                    bb_full_list,
                    ciou_full_list,
                    confidence_full_list,
                )
                subset_name = (
                    f"{subset}@{int(iou_thr*100)}"
                    if subset is not None
                    else f"@{int(iou_thr*100)}"
                )
                if len(ciou_list) == 0:
                    p, r, ap, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    p, r = self.calc_precision_recall(
                        bb_list, ciou_list, conf_list, -1000, iou_thr
                    )
                    ap = self.calc_ap(bb_list, ciou_list, conf_list, iou_thr)
                    auc = self.cal_auc(bb_list, ciou_list)

                    conf_thr = list(sorted(conf_list))[:: max(1, len(conf_list) // 10)]
                    pr = [
                        self.calc_precision_recall(
                            bb_list, ciou_list, conf_list, thr, iou_thr
                        )
                        for thr in conf_thr
                    ]
                    f1 = [2 * r * p / (r + p) if r + p > 0 else 0.0 for p, r in pr]
                metrics[f"Precision-{subset_name}"] = p * 100
                # metrics[f'Recall-{subset_name}'] = r
                if np.isnan(f1).any():
                    metrics[f"F1-{subset_name}"] = f1
                else:
                    metrics[f"F1-{subset_name}"] = " ".join(
                        [f"{f*100:.1f}" for f in f1]
                    )
                metrics[f"AP-{subset_name}"] = ap * 100
                metrics[f"AUC-{subset_name}"] = auc * 100

        return metrics

    def gather_results(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            return (
                self.name_list,
                self.area_list,
                self.bb_list,
                self.ciou_list,
                self.confidence_list,
            )
        world_size = dist.get_world_size()

        bb_list = [None for _ in range(world_size)]
        dist.all_gather_object(bb_list, self.bb_list)
        bb_list = [x for bb in bb_list for x in bb]

        area_list = [None for _ in range(world_size)]
        dist.all_gather_object(area_list, self.area_list)
        area_list = [x for area in area_list for x in area]

        ciou_list = [None for _ in range(world_size)]
        dist.all_gather_object(ciou_list, self.ciou_list)
        ciou_list = [x for ciou in ciou_list for x in ciou]

        confidence_list = [None for _ in range(world_size)]
        dist.all_gather_object(confidence_list, self.confidence_list)
        confidence_list = [x for conf in confidence_list for x in conf]

        name_list = [None for _ in range(world_size)]
        dist.all_gather_object(name_list, self.name_list)
        name_list = [x for name in name_list for x in name]

        return name_list, area_list, bb_list, ciou_list, confidence_list

    def precision_at_50(self):
        ss = [i for i, bb in enumerate(self.bb_list) if bb > 0]
        return np.mean(np.array([self.ciou_list[i] for i in ss]) > 0.5)

    def precision_at_50_object(self):
        max_num_obj = max(self.bb_list)
        for num_obj in range(1, max_num_obj + 1):
            ss = [i for i, bb in enumerate(self.bb_list) if bb == num_obj]
            precision = np.mean(np.array([self.ciou_list[i] for i in ss]) > 0.5)
            print("\n" + f"num_obj:{num_obj}, precision:{precision}")

    def f1_at_50(self):
        # conf_thr = np.array(self.confidence_list).mean()
        p, r = self.calc_precision_recall(
            self.bb_list,
            self.ciou_list,
            self.confidence_list,
            self.default_conf_thr,
            0.5,
        )
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def ap_at_50(self):
        return self.calc_ap(self.bb_list, self.ciou_list, self.confidence_list, 0.5)

    def clear(self):
        self.ciou_list = []
        self.area_list = []
        self.confidence_list = []
        self.name_list = []
        self.bb_list = []

    def update(self, bb, gt, conf, pred, pred_thr, name):
        infer = torch.argmax(pred, dim=0).cpu().numpy()
        # infer = np.zeros((224, 224))
        # infer[pred.cpu().numpy() >= pred_thr] = 1
        # Compute ciou between prediction and ground truth
        ciou = np.sum(infer * gt) / (np.sum(gt) + np.sum(infer * (gt == 0)))

        # Compute ground truth size
        area = gt.sum()

        # Save
        self.confidence_list.append(conf)
        self.ciou_list.append(ciou)
        self.area_list.append(area)
        self.name_list.append(name)
        self.bb_list.append(len(bb))

    def display_results(self, local_rank):
        iou_th =  int(self.iou_thrs[0]*100)
        metrics = self.finalize_stats()
        display_title = "Detection Performance on VGGSound"
        display_data = [
            [
                f"AP@Iou({iou_th})",
                f"F1@Iou({iou_th})",
                f"Prec@Iou({iou_th})",
                f"AUC@Iou({iou_th})",
                f"AP(Huge)",
                f"AP(Large)",
                f"AP(Medium)",
                f"AP(Small)",
            ],
        ]
        ap50 = metrics[f"AP-all@{iou_th}"]
        f1_max = np.asarray(metrics[f"F1-all@{iou_th}"].split(" ")).astype(float)
        f1_max = max(f1_max)
        prec50 = metrics[f"Precision-all@{iou_th}"]
        auc50 = metrics[f"AUC-all@{iou_th}"]
        ap50_huge = metrics[f"AP-huge@{iou_th}"]
        ap50_large = metrics[f"AP-large@{iou_th}"]
        ap50_medium = metrics[f"AP-medium@{iou_th}"]
        ap50_small = metrics[f"AP-small@{iou_th}"]

        display_data.append(
            [
                "{:.4f}".format(ap50),
                "{:.4f}".format(f1_max),
                "{:.4f}".format(prec50),
                "{:.4f}".format(auc50),
                "{:.4f}".format(ap50_huge),
                "{:.4f}".format(ap50_large),
                "{:.4f}".format(ap50_medium),
                "{:.4f}".format(ap50_small),
            ]
        )

        table = AsciiTable(display_data, display_title)
        table.justify_columns[-1] = "right"
        table.inner_footing_row_border = True
        if local_rank <= 0:
            logger.success("\n{}".format(table.table))
            return (
                ap50,
                f1_max,
                prec50,
                auc50,
                ap50_huge,
                ap50_large,
                ap50_medium,
                ap50_small,
            )
        else:
            return 0, 0, 0, 0, 0, 0, 0, 0

    def save_results(self):
        name_list, area_list, bb_list, ciou_list, conf_list = self.gather_results()
        save_results(
            name_list,
            area_list,
            bb_list,
            ciou_list,
            conf_list,
            os.path.join(self.results_save_dir, f"sample_cious.txt"),
        )

        metrics = self.finalize_stats()
        open(os.path.join(self.results_save_dir, f"metrics.txt"), "w").write(
            "\n".join(
                [
                    f"{k}: {metrics[k]}"
                    for k in sorted(metrics.keys())
                    if metrics[k] is not np.nan
                ]
            )
        )

    def make_table(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            # print ('[RESULTS] Performance on ActivityNet detection task.')
            # print ('Average-mAP: {}'.format(self.average_mAP))
            # print ('Average-mAP: {}\n'.format(self.mAP))

            display_title = "Detection Performance on VGGSound"
            display_data = [["IoU thresh"], ["mAP"]]

            for i in range(len(self.tiou_thresholds)):
                display_data[0].append("{:.02f}".format(self.tiou_thresholds[i]))
                display_data[1].append("{:.04f}".format(self.mAP[i]))
            display_data[0].append("Average")
            display_data[1].append("{:.04f}".format(self.average_mAP))
            table = AsciiTable(display_data, display_title)
            table.justify_columns[-1] = "right"
            table.inner_footing_row_border = True
            print(table.table)


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def visualize(raw_image, boxes):
    import cv2

    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:

        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam_v2(model, args):
    # optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    imgnet = []
    others = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "imgnet" in name:
                imgnet.append(param)
            else:
                others.append(param)

    optimizer = Adam(
        [{"params": imgnet}, {"params": others}],
        lr=args.init_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(
        optimizer_grouped_parameters, 
        lr=args.init_lr, 
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(
        optimizer_grouped_parameters,
        lr=args.init_lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=args.step_size, gamma=0.1
    # )
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode="w", encoding="utf-8") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def save_results(name_list, area_list, bb_list, iou_list, conf_list, filename):
    with open(filename, "w") as file_iou:
        file_iou.write("name,area,bb,ciou,conf\n")
        for indice in np.argsort(iou_list):
            file_iou.write(
                f"{name_list[indice]},{area_list[indice]},{bb_list[indice]},{iou_list[indice]},{conf_list[indice]}\n"
            )


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    wu = 0 if "warmup_epochs" not in vars(args) else args.warmup_epochs
    if args.lr_schedule == "cos":  # cosine lr schedule
        if epoch < wu:
            lr = args.init_lr * epoch / wu
        else:
            lr = (
                args.init_lr
                * 0.5
                * (1.0 + math.cos(math.pi * (epoch - wu) / (args.epochs - wu)))
            )
    elif args.lr_schedule == "cte":  # constant lr
        lr = args.init_lr
    else:
        raise ValueError

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(
            m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2_5D_depth):
            group_decay.append(m.weight_0)
            group_decay.append(m.weight_1)
            group_decay.append(m.weight_2)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2_5D_disp):
            group_decay.append(m.weight_0)
            group_decay.append(m.weight_1)
            group_decay.append(m.weight_2)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif (
            isinstance(m, norm_layer)
            or isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
            or isinstance(m, nn.GroupNorm)
            or isinstance(m, nn.LayerNorm)
        ):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)
        elif isinstance(m, nn.Embedding):
            group_decay.append(m)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    return weight_group
