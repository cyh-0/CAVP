import argparse


def add_tag(tags, key):
    if len(tags) != 0:
        tags.append(key)
    else:
        tags = [key]
    return tags


def load_args_and_config():
    parser = argparse.ArgumentParser(description="Audio-Visual Recognition")
    # default parameters for dgx.
    parser.add_argument("--pvc", action="store_true", help="pvc or not")
    parser.add_argument("--dgx", action="store_true", help="dgx or not")
    parser.add_argument(
        "--wandb_mode", default="online", type=str, help="Mode of wandb API"
    )
    parser.add_argument(
        "--wandb_dir",
        default="/mnt/beegfs/mccarthy/backed_up/projects/braix/ychen/wandb_logs/CAVP",
        type=str,
        help="Mode of wandb API",
    )
    parser.add_argument("--tags", nargs="+", default="")
    parser.add_argument("--run_note", default="", type=str, help="Notes for run")
    parser.add_argument(
        "--experiment_name",
        default="ca+dp_ctr",
        type=str,
        help="Mode of wandb API",
    )
    # hyp-parameters parameters for hardware
    parser.add_argument("--gpus", default=1, type=int, help="gpus in use")
    parser.add_argument("--nodes", default=1, type=int, help="distributed or not")
    parser.add_argument("--local_rank", default=0, type=int, help="distributed or not")
    # Train
    parser.add_argument("--num_workers", default=8, type=int, help="Batch Size")
    # Model
    parser.add_argument("--num_queries", default=100, type=int, help="Batch Size")
    parser.add_argument("--visual_backbone", type=int, default=50)
    parser.add_argument("--seg_model", type=str, default="DeepLabV3Plus")
    parser.add_argument("--use_baseline", default=False, action="store_true")
    # Data
    parser.add_argument("--semi_ratio", default="1/1", type=str)
    parser.add_argument("--setup", default="coco", type=str)
    parser.add_argument("--use_synthetic", default=False, action="store_true")
    # Flags
    parser.add_argument("--cavp_flag", default=False, action="store_true")
    parser.add_argument("--cutmix_flag", default=False, action="store_true")
    # hyp-parameters for function.
    parser.add_argument("--batch_size", default=16, type=int, help="Batch Size")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=60, type=int)
    # Mode
    parser.add_argument("--ignore_ckpt", default=False, action="store_true")
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--use_multi_source", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--ow_rate", default=0.5, type=float)
    #
    parser.add_argument("--avsbench_split", default="all", type=str)

    args = parser.parse_args()
    args.visual_backbone_pretrain_path = "../ckpts/pretrained/resnet{}.pth".format(
        args.visual_backbone
    )

    """ Configs """
    if args.setup == "vpo_ss":
        from config.config_vpo_ss import config
        args.tags = add_tag(args.tags, "vpo_ss")
    elif args.setup == "vpo_ms":
        from config.config_vpo_ms import config
        args.tags = add_tag(args.tags, "vpo_ms")
    elif args.setup == "vpo_msmi":
        from config.config_vpo_msmi import config
        args.tags = add_tag(args.tags, "vpo_msmi")
    elif args.setup == "avss":
        from config.config_avss import config
    else:
        raise ValueError("Unknow setup")

    if args.local:
        args.wandb_dir = "./"

    return args, config
