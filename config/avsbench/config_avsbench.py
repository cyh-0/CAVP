import os
from easydict import EasyDict
from config.class_list import index_table_avs

C = EasyDict()
config = C
cfg = C

C.LAMBDA_1 = 50

##############################
# TRAIN
# C.BACKBONE = 50
C.TRAIN = EasyDict()
# TRAIN.SCHEDULER
C.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
C.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "ckpts/avsbench/vggish-10086976.pth"
C.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
C.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
C.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "ckpts/avsbench/vggish_pca_params-970ea276.pth"
C.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
C.TRAIN.PRETRAINED_RESNET50_PATH = "ckpts/avsbench/resnet50-19c8e357.pth"
C.TRAIN.PRETRAINED_RESNET101_PATH = "ckpts/avsbench/resnet101-cd907fc2.pth"
C.TRAIN.PRETRAINED_PVTV2_PATH = "ckpts/avsbench/pvt_v2_b5.pth"
