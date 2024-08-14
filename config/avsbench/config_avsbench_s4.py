import os
from easydict import EasyDict as edict
from config.class_list import index_table_avs

cfg = edict()

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "./avsbench_data/Single-source/s4_meta_data.csv"
cfg.DATA.DIR_IMG = "./avsbench_data/Single-source/s4_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "./avsbench_data/Single-source/s4_data/audio_log_mel"
cfg.DATA.DIR_AUDIO_WAV = "./avsbench_data/Single-source/s4_data/audio_wav"
cfg.DATA.DIR_MASK = "./avsbench_data/Single-source/s4_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
###############################

def get_cfg(data_root):
    cfg.DATA.ANNO_CSV = os.path.join(data_root, cfg.DATA.ANNO_CSV)
    cfg.DATA.DIR_IMG = os.path.join(data_root, cfg.DATA.DIR_IMG)
    cfg.DATA.DIR_AUDIO_LOG_MEL = os.path.join(data_root, cfg.DATA.DIR_AUDIO_LOG_MEL)
    cfg.DATA.DIR_AUDIO_WAV = os.path.join(data_root, cfg.DATA.DIR_AUDIO_WAV)
    cfg.DATA.DIR_MASK = os.path.join(data_root, cfg.DATA.DIR_MASK)
    return cfg
