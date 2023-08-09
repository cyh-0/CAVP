import argparse
import os
import random
import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from easydict import EasyDict
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from engine.engine import Engine
from engine.utils import group_weight
from engine.lr_policy import WarmUpPolyLR
from models.audio.audio_network import AudioModel
from models.visual.visual_network import VisualModel
from loguru import logger
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
    # bkb: [:2]
    param_lists_v = group_weight(
        param_lists_v,
        model_v.resnet,
        norm_layer=torch.nn.BatchNorm2d,
        lr=hyp_param_.lr,
    )
    # [2:18]
    for module in model_v.backbone_layer:
        param_lists_v = group_weight(
            param_lists_v, module, torch.nn.BatchNorm2d, hyp_param_.lr * 1.0
        )
    # [18:]
    for module in model_v.business_layer:
        param_lists_v = group_weight(
            param_lists_v, module, torch.nn.BatchNorm2d, hyp_param_.lr * 10.0
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
        hyp_param_.coco_num_classes if hyp_param_.use_coco else hyp_param_.num_classes
    )

    from models.visual.avsbench import ResNet_AVSModel as AVSModel
    from config.avsbench.config_avsbench import cfg as avs_cfg

    hyp_param_.update(**vars(avs_cfg))

    model_v = AVSModel.Pred_endecoder(
        channel=256,
        config=hyp_param_,
        tpavi_stages=[0, 1, 2, 3],
        tpavi_vv_flag=False,
        tpavi_va_flag=True,
    )
    model_a = AudioModel(
        "vgg",
        hyp_param_.audio_backbone_pretrain_path,
        out_plane=128,
    )

    num_param = sum(p.numel() for p in model_v.parameters())
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

    import pandas

    if hyp_param_.use_coco:
        df_name_ = (
            "coco_av_train.csv"
            if hyp_param_.setup == "coco_ss"
            else "coco_multi_source.csv"
        )
        csv_path = os.path.join(hyp_param_.coco_data_path, df_name_)
    else:
        df_name_ = "data_synthetic.csv" if hyp_param_.use_synthetic else "data.csv"
        csv_path = os.path.join(hyp_param_.data_path, df_name_)

    logger.warning(f"Using <<{df_name_}>>")
    csv_ = pandas.read_csv(csv_path)
    if hyp_param_.setup == "coco_ms":
        from dataset.multi_source.av_datasets import AudioVisualDataset
        from dataset.multi_source.visual.visual_dataset import prepare_train_data
    else:
        from dataset.single_source.av_datasets import AudioVisualDataset
        from dataset.single_source.visual.visual_dataset import prepare_train_data
    if hyp_param_.use_coco:
        csv_ = prepare_train_data(csv_.copy(), hyp_param_)

    hyp_param_.audio_len = 1

    train_dataset = AudioVisualDataset(
        args=hyp_param_, mode="train", dataframe=csv_[csv_["split"] == "train"]
    )
    # val_dataset = AudioVisualDataset(
    #     args=hyp_param_, mode="test", dataframe=csv_[csv_["split"] == "val"]
    # )
    test_dataset = AudioVisualDataset(
        args=hyp_param_, mode="test", dataframe=csv_[csv_["split"] == "test"]
    )
    # val_csv_ = csv_[csv_['split'] == 'val']
    final_batch_size = hyp_param_.batch_size * hyp_param_.gpus
    lr_policy = WarmUpPolyLR(
        hyp_param_.lr,
        hyp_param_.lr_power,
        int(len(train_dataset) / final_batch_size) * hyp_param_.epochs,
        len(train_dataset) // final_batch_size * hyp_param_.warm_up_epoch,
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if hyp_param_.ddp
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyp_param_.batch_size,
        shuffle=(train_sampler is None),
        num_workers=hyp_param_.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=hyp_param_.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    from trainer.avs_trainer import BASELINE

    trainer = BASELINE(
        hyp_param_,
        train_loader,
        engine=engine,
        visual_tool=wandb_,
        lr_scheduler=lr_policy,
    )

    # model_v.module.load_state_dict(torch.load("ckpts/t.pth")['visual_model'], strict=True)
    # model_a.module.load_state_dict(torch.load("ckpts/t.pth")['audio_model'], strict=True)

    for epoch in range(0, hyp_param_.epochs):
        engine.register_state(
            dataloader=train_loader,
            model_v=model_v,
            optimizer_v=optimizer_v,
            model_a=model_a,
            optimizer_a=optimizer_a,
        )
        if hyp_param_.ddp:
            train_loader.sampler.set_epoch(epoch)
        trainer.train(model_v, model_a, optimizer_v, optimizer_a, epoch, train_loader)

        if local_rank <= 0:
            trainer.validation(model_v, model_a, epoch, test_loader)
        ddp_utils.barrier(hyp_param_.ddp)

    if hyp_param_.local_rank <= 0:
        wandb_.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Visual Recognition")
    # default parameters for dgx.
    parser.add_argument("--pvc", action="store_true", help="pvc or not")
    parser.add_argument("--dgx", action="store_true", help="dgx or not")
    parser.add_argument(
        "--wandb_mode", default="online", type=str, help="Mode of wandb API"
    )
    parser.add_argument("--tags", nargs="+", default="")
    parser.add_argument("--run_note", default="", type=str, help="Notes for run")
    parser.add_argument(
        "--experiment_name",
        default="ca+dp_ctr",
        type=str,
        help="Mode of wandb API",
    )

    # hyp-parameters parameters for hardware.
    parser.add_argument("--gpus", default=1, type=int, help="gpus in use")
    parser.add_argument("--nodes", default=1, type=int, help="distributed or not")
    parser.add_argument("--local_rank", default=0, type=int, help="distributed or not")
    # Train
    parser.add_argument("--num_workers", default=8, type=int, help="Batch Size")
    # Model
    parser.add_argument("--visual_backbone", type=int, default=50)
    parser.add_argument("--seg_model", type=str, default="DeepLabV3Plus")
    parser.add_argument("--use_baseline", default=False, action="store_true")
    # Data
    parser.add_argument("--setup", default="coco", type=str)
    parser.add_argument("--use_synthetic", default=False, action="store_true")
    # hyp-parameters for function.
    parser.add_argument("--batch_size", default=16, type=int, help="Batch Size")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=60, type=int)
    # Mode
    parser.add_argument("--use_multi_source", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()

    """ Configs """
    if args.setup == "coco_ss":
        from config.config_coco_ss import config

        args.tags = add_tag(args.tags, "coco_ss")
    elif args.setup == "coco_ms":
        from config.config_coco_ms import config

        args.tags = add_tag(args.tags, "coco_ms")
    elif args.setup == "coco_org":
        from config.config_coco_org import config

        args.tags = add_tag(args.tags, "coco_org")
    elif args.setup == "avs":
        from config.config_avs import config
    else:
        raise ValueError("Unknow setup")

    hyp_param = EasyDict(config)
    # hyp_param = EasyDict({**vars(args), **config})
    hyp_param.update(**vars(args))
    # adjust value for multi-gpus training.
    hyp_param.lr *= hyp_param.gpus
    hyp_param.ddp = True if hyp_param.gpus > 1 else False
    hyp_param.world_size = hyp_param.gpus * hyp_param.nodes

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
        hyp_param.image_width = 128
        hyp_param.image_height = 128

    logger.critical(f"SETUP: {hyp_param.setup}")
    logger.critical(f"EPOCH: {hyp_param.epochs}")
    logger.critical(f"BACKBONE: {hyp_param.visual_backbone}")
    logger.critical(f"BATCH SIZE: {hyp_param.batch_size}")
    logger.critical(f"LR: {hyp_param.lr}")
    logger.critical(f"WEIGHT DECAY: {hyp_param.weight_decay}")

    if hyp_param.ddp:
        mp.spawn(main, nprocs=hyp_param.gpus, args=(hyp_param.gpus, hyp_param))
    else:
        main(0, hyp_param.gpus, hyp_param)
