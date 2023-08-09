import os
from easydict import EasyDict
from config.class_list import index_table_avs


cfg = EasyDict()
cfg.BATCH_SIZE = 4 # default 4
cfg.LAMBDA_1 = 0.5 # default: 0.5
cfg.MASK_NUM = 10 # 10 for fully supervised
cfg.NUM_CLASSES = 71 # 70 + 1 background

###############################
# DATA
cfg.DATA = EasyDict()
cfg.DATA.CROP_IMG_AND_MASK = True
cfg.DATA.CROP_SIZE = 224 # short edge

cfg.DATA.META_CSV_PATH = "../audio_visual/avsbench_semantic/metadata.csv" #! notice: you need to change the path
cfg.DATA.LABEL_IDX_PATH = "../audio_visual/avsbench_semantic/label2idx.json" #! notice: you need to change the path

cfg.DATA.DIR_BASE = "../audio_visual/avsbench_semantic" #! notice: you need to change the path
cfg.DATA.DIR_MASK = "../audio_visual/avsbench_semantic/v2_data/gt_masks" #! notice: you need to change the path
cfg.DATA.DIR_COLOR_MASK = "../audio_visual/avsbench_semantic/v2_data/gt_color_masks_rgb" #! notice: you need to change the path
cfg.DATA.IMG_SIZE = (224, 224)
###############################
cfg.DATA.RESIZE_PRED_MASK = True
cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (360, 240) # (width, height)