import os
from easydict import EasyDict
from config.class_list import index_table_coco, coco_class_dict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Image Settings"""
C.image_width = 512  # 224 # 512
C.image_height = 512  # 224 # 512
C.image_mean = [0.485, 0.456, 0.406]
C.image_std = [0.229, 0.224, 0.225]

"""Audio Settings"""
C.audio_len = 3.0
# C.load_audio_feature = False
C.spec_min = -100
C.spec_max = 100
C.audio_mean = [0.0]
C.audio_std = [12.0]

"""Root Directory Config"""
C.repo_name = "vsseg"
C.root_dir = os.path.realpath("")

"""Data Dir and Weight Dir"""
C.root_dataset_dir = "../audio_visual"
C.dataset_name = "avsbench_data_single_plus/"
C.vgg_root = "vggsound_bench/VGGSound"
C.data_path = os.path.join(C.root_dataset_dir, C.dataset_name)
C.vgg_data_path = os.path.join(C.root_dataset_dir, C.vgg_root)
C.synth_data_path = os.path.join(C.root_dataset_dir, C.vgg_root)

"""COCO setup"""
C.use_vpo = True
C.replace_name = False
C.index_table = index_table_coco
C.class_dict = coco_class_dict
C.coco_root = "VPO/VPO-SS/"
C.vpo_num_classes = 21 + 1
C.vpo_data_path = os.path.join(C.root_dataset_dir, C.coco_root)
C.coco_img_root = os.path.join(C.vpo_data_path, "data")
C.coco_mask_root = os.path.join(C.vpo_data_path, "mask")

"""Model Settings"""
C.visual_backbone = 101
C.last_three_dilation_stride = [False, True, True]
C.audio_backbone = "vgg" if C.audio_len == 1 else "18"
C.visual_backbone_pretrain_path = "../ckpts/pretrained/resnet{}.pth".format(
    C.visual_backbone
)
C.audio_backbone_pretrain_path = "../ckpts/pretrained/vgg.pth"

"""Optimisation Settings"""
C.lr = 1e-3  # 1e-3 for binary
C.lr_power = 0.9
C.batch_size = 16
C.epochs = 80
C.momentum = 0.9
C.weight_decay = 5e-4
C.num_classes = 24
C.warm_up_epoch = 0
C.num_workers = 8
C.ciou_thre = [0.3]
C.pred_thre = 0.4

"""Wandb Config"""
# Specify you wandb environment KEY; and paste here
C.wandb_key = "bf10181288fa64afd20c86feac6ea4b1abfad71e"
# Your project [work_space] name
C.proj_name = "VPO"
C.experiment_name = "baseline+audio(pretrain)"

# half pretrained_ckpts-loader upload images; loss upload every iteration
# C.upload_image_step = [0, int((C.num_train_imgs / C.batch_size) / 2)]

C.display_iter = 1
C.upload_iter = 100

# False for debug; True for visualize
C.wandb_mode = "online"
C.wandb_dir = "/mnt/beegfs/mccarthy/backed_up/projects/braix/ychen/wandb_logs/CAVP-P"

# """Save Config"""
# C.saved_dir = os.path.join("/media/data/yy/ckpts/avseg", C.experiment_name)
# if not os.path.exists(C.saved_dir):
#     os.mkdir(C.saved_dir)
