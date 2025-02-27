import argparse
import os
import random
import time

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from easydict import EasyDict
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config.flags import add_tag, load_args_and_config
from engine.engine import Engine
from engine.lr_policy import WarmUpPolyLR
from engine.utils import group_weight
from models.audio.audio_network import AudioModel
from models.visual.visual_network import VisualModel
from utils import ddp_utils


def add_tag(tags, key):
    if len(tags) != 0:
        tags.append(key)
    else:
        tags = [key]
    return tags


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def set_group_lr(model_v, hyp_param_):
    param_lists_v = []
    # 8
    for module in model_v.segment.business_layer:
        param_lists_v = group_weight(
            param_lists_v, module, torch.nn.BatchNorm2d, hyp_param_.lr * 10.0
        )
    #
    param_lists_v = group_weight(
        param_lists_v,
        model_v.backbone,
        norm_layer=torch.nn.BatchNorm2d,
        lr=hyp_param_.lr,
    )
    if not hyp_param_.use_baseline:
        param_lists_v.append(
            {"params": model_v.visual_projector.parameters(), "lr": hyp_param_.lr * 1}
        )
        param_lists_v.append(
            {"params": model_v.cross_att.parameters(), "lr": hyp_param_.lr * 1}
        )

    return param_lists_v


def main(local_rank, ngpus_per_node, hyp_param_):
    
    hyp_param_.local_rank = local_rank
    engine = Engine(custom_arg=hyp_param_, logger=logger)
    ddp_utils.supress_printer(hyp_param_.ddp, local_rank)

    if hyp_param_.local_rank <= 0:
        from utils.tensor_board import Tensorboard

        wandb_ = Tensorboard(hyp_param_)
    else:
        wandb_ = None

    hyp_param_.num_classes = (
        hyp_param_.vpo_num_classes if hyp_param_.use_vpo else hyp_param_.num_classes
    )

    if hyp_param_.use_baseline:
        model_v = VisualModel(
            hyp_param_.visual_backbone,
            hyp_param_.visual_backbone_pretrain_path,
            num_classes=hyp_param_.num_classes,
            seg_model=hyp_param_.seg_model,
            last_three_dilation_stride=hyp_param_.last_three_dilation_stride,
        )
        model_a = AudioModel(
            hyp_param_.audio_backbone,
            hyp_param_.audio_backbone_pretrain_path,
            out_plane=2048 if hyp_param_.visual_backbone == 50 else 512,
        )
    else:
        from models.cavp_model import CAVP

        model_v = CAVP(
            hyp_param_.visual_backbone,
            hyp_param_.visual_backbone_pretrain_path,
            num_classes=hyp_param_.num_classes,
            audio_backbone_pretrain_path=hyp_param_.audio_backbone_pretrain_path,
            visual_backbone=hyp_param_.visual_backbone,
            args=hyp_param_,
        )
        model_a = model_v.audio_backbone

    num_param = sum(p.numel() for p in model_v.parameters() if p.requires_grad)
    MODEL_PARAMS = numpy.round(num_param / 1e6, 4)
    logger.warning("Number of trainable parameters: {}M".format(MODEL_PARAMS))
    if local_rank <= 0:
        wandb_.tensor_board.config.update({"MODEL_PARAMS": MODEL_PARAMS})

    param_lists_v = set_group_lr(model_v, hyp_param_)

    optimizer_v = torch.optim.SGD(
        param_lists_v,
        lr=hyp_param_.lr,
        momentum=hyp_param_.momentum,
        weight_decay=hyp_param_.weight_decay,
    )

    optimizer_a = torch.optim.Adam(params=model_a.parameters(), lr=hyp_param_.lr)

    if hyp_param_.ddp:
        torch.cuda.set_device(hyp_param_.local_rank)
        model_v.cuda(hyp_param_.local_rank)
        visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(model_v)
        model_v = DDP(
            visual_model,
            device_ids=[hyp_param_.local_rank],
            find_unused_parameters=True,
        )

        model_a.cuda(hyp_param_.local_rank)
        model_a = nn.SyncBatchNorm.convert_sync_batchnorm(model_a)
        model_a = DDP(
            model_a, device_ids=[hyp_param_.local_rank], find_unused_parameters=True
        )
    else:
        model_v = nn.DataParallel(model_v, device_ids=["cuda:0"])
        model_a = nn.DataParallel(model_a, device_ids=["cuda:0"])


    from dataset.avss.avss_datasets import AVSSDataset
    

    train_dataset = AVSSDataset(args=hyp_param_, mode="train")
    test_dataset = AVSSDataset(args=hyp_param_, mode="test")
    wandb_.pallete = train_dataset.dataset_v.pallete
    
    # Data
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if hyp_param_.ddp
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyp_param_.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=hyp_param_.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=hyp_param_.num_workers,
        pin_memory=True,
    )


    from trainer.trainer_cavp_avss_image import CAVP_TRAINER

    final_batch_size = hyp_param_.batch_size * hyp_param_.gpus
    lr_policy = WarmUpPolyLR(
        hyp_param_.lr,
        hyp_param_.lr_power,
        int(len(train_dataset) / final_batch_size) * hyp_param_.epochs,
        len(train_dataset) // final_batch_size * hyp_param_.warm_up_epoch,
    )
    trainer = CAVP_TRAINER(
        hyp_param_,
        train_loader,
        engine=engine,
        visual_tool=wandb_,
        lr_scheduler=lr_policy,
    )

    ckpt = torch.load("./avss_224.pth")['model']
    model_v.load_state_dict(ckpt, strict=False)

    trainer.validation(model_v, model_a, -1, test_loader)

    if hyp_param_.local_rank <= 0:
        wandb_.finish()


if __name__ == "__main__":
    logger.warning("RUNNING AVSS")
    args, config = load_args_and_config()
    hyp_param = EasyDict(config)
    hyp_param.update(**vars(args))
    # adjust value for multi-gpus training.
    hyp_param.lr *= hyp_param.gpus
    hyp_param.ddp = True if hyp_param.gpus > 1 else False
    hyp_param.world_size = hyp_param.gpus * hyp_param.nodes

    if hyp_param.avsbench_split == "all":
        hyp_param.num_classes = 71

    """ Pretrains """
    if hyp_param.seg_model == "HRNet":
        hyp_param.visual_backbone = "HRNet-W48"
    elif hyp_param.seg_model == "OCR":
        hyp_param.visual_backbone = "HRNet-W48"

    seed_it(hyp_param.seed + hyp_param.local_rank)

    if args.debug:
        logger.critical("DEBUG MODE ACTIVATED")
        hyp_param.wandb_mode = "disabled"
        hyp_param.experiment_name = "dummpy_test"
        # hyp_param.image_width = 128
        # hyp_param.image_height = 128

    logger.warning(f"SETUP: {hyp_param.setup}")
    logger.warning(f"EPOCH: {hyp_param.epochs}")
    logger.warning(f"BACKBONE: {hyp_param.visual_backbone}")
    logger.warning(f"BATCH SIZE: {hyp_param.batch_size}")
    logger.warning(f"LR: {hyp_param.lr}")
    logger.warning(f"WEIGHT DECAY: {hyp_param.weight_decay}")

    # if hyp_param.resize_flag:
    #     logger.critical(" *** RESIZE FLAG IS ON ***")
    #     hyp_param.image_width = 224
    #     hyp_param.image_height = 224
    #     logger.critical(f"RESIZE: {hyp_param.image_width}x{hyp_param.image_height}")
    #     hyp_param.num_classes = 2

    if hyp_param.ddp:
        mp.spawn(main, nprocs=hyp_param.gpus, args=(hyp_param.gpus, hyp_param))
    else:
        main(0, hyp_param.gpus, hyp_param)