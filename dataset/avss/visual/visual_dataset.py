import os
import torch, torchvision
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
from loguru import logger
from config.avss.config_avsbench_semantics import cfg as avss_cfg
from engine.utils import DeNormalize
from utils.tensor_board import colorize_mask, pil_image_grid

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    # v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(np.array(v2_pallete).reshape(-1, 3)) == len(label_to_pallete_idx)
    return v2_pallete

def get_fn(x, type="image"):
    if type == "audio":
        fn = os.path.join(avss_cfg.DATA.DIR_BASE, x['label'], x['uid'], 'audio.wav')
    elif type == "image":
        fn = os.path.join(avss_cfg.DATA.DIR_BASE, x['label'], x['uid'], 'frames')
    elif type == "mask":
        fn = os.path.join(avss_cfg.DATA.DIR_BASE, x['label'], x['uid'], 'labels_semantic')
    return fn

def prepare_data(df):
    ##########################
    df["audio_fp"] = df.apply(lambda x: get_fn(x, type="audio"), axis=1)
    df["image_fp"] = df.apply(lambda x: get_fn(x, type="image"), axis=1)
    df["mask_fp"] = df.apply(lambda x: get_fn(x, type="mask"), axis=1)
    return df

class VisualDataset(Dataset):
    def __init__(self, data_frame, mode, transform=None, args=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.df = prepare_data(data_frame)
        self.name_list = self.df["uid"].tolist()
        self.frame_list = self.df["image_fp"].tolist()
        self.label_list = self.df["mask_fp"].tolist()
        self.category_list = self.df["a_obj"].tolist()
        self.num_classes = avss_cfg.NUM_CLASSES
        with open(avss_cfg.DATA.LABEL_IDX_PATH, 'r') as fr:
            self.index_table = json.load(fr)

        logger.info("{} videos are used for {}.".format(len(self.df), self.mode))
        self.pallete = get_v2_pallete(avss_cfg.DATA.LABEL_IDX_PATH, num_cls=avss_cfg.NUM_CLASSES)
        self.denorm = DeNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        self.resize_flag = args.resize_flag
        self.avsbench_split = args.avsbench_split
        if self.avsbench_split != "all":
            logger.critical(f"CONVERT LABEL TO BINARY")


    def __getflag(self, set):
        if set == 'v1s': # data from AVSBench-object single-source subset (5s, gt is only the first annotated frame)
            vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
            gt_temporal_mask_flag  = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])#.bool()
        elif set == 'v1m': # data from AVSBench-object multi-sources subset (5s, all 5 extracted frames are annotated)
            vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
            gt_temporal_mask_flag  = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
        elif set == 'v2': # data from newly collected videos in AVSBench-semantic (10s, all 10 extracted frames are annotated))
            vid_temporal_mask_flag = torch.ones(10)#.bool()
            gt_temporal_mask_flag = torch.ones(10)#.bool()
        return vid_temporal_mask_flag, gt_temporal_mask_flag

    def __get_image(self, fn_img):
        img_path_list = sorted(os.listdir(fn_img)) # 5 for v1, 10 for new v2
        imgs_num = len(img_path_list)
        imgs_pad_zero_num = 10 - imgs_num
        imgs = []
        for img_id in range(imgs_num):
            img_path = os.path.join(fn_img, "%d.jpg"%(img_id))
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)

        for pad_i in range(imgs_pad_zero_num):
            img = Image.fromarray(np.zeros_like(np.asarray(img)))
            imgs.append(img)
        return imgs

    def __get_mask(self, fn_label, subset):
        labels = []
        mask_path_list = sorted(os.listdir(fn_label))
        for mask_path in mask_path_list:
            if not mask_path.endswith(".png"):
                mask_path_list.remove(mask_path)
        mask_num = len(mask_path_list)
        if self.mode != 'train':
            if subset == 'v2':
                assert mask_num == 10
            else:
                assert mask_num == 5

        mask_num = len(mask_path_list)
        label_pad_zero_num = 10 - mask_num
        for mask_id in range(mask_num):
            mask_path = os.path.join(fn_label, "%d.png"%(mask_id))
            mask_ = Image.open(mask_path)
            labels.append(mask_)
        for pad_j in range(label_pad_zero_num):
            mask_ = Image.fromarray(np.zeros_like(np.asarray(mask_)))
            labels.append(mask_)
        return labels

    def __check(self, image, label): 
        torchvision.utils.save_image(self.denorm(torch.stack(list(image))), "test.png", nrow=5)
        pil_image_grid([colorize_mask(item.numpy(), self.pallete) for item in label], 2, 5)
        
    def __getitem__(self, idx):
        uid = self.name_list[idx]
        df_curr = self.df[self.df.uid == uid].iloc[0]
        subset = df_curr["label"]
        #
        fn_img = self.frame_list[idx]
        fn_label = self.label_list[idx]
        frame_available, mask_available = self.__getflag(subset)
        images = self.__get_image(fn_img)
        labels = self.__get_mask(fn_label, subset=subset)
        # image = Image.open(fn_img).convert("RGB")
        # label = Image.open(fn_label)/
        pack_ = [self.transform(image, label) for image, label in zip(images, labels)]
        image, label = list(zip(*pack_))
        image = torch.stack(list(image))
        label = torch.stack(list(label))

        class_label = [F.one_hot(torch.unique(label[i][label[i]!=255]), num_classes=self.num_classes).sum(0) for i in range(len(label))]
        class_label = torch.stack(class_label)

        if self.resize_flag:
            if self.avsbench_split != "all":
                label[(label != 255) & (label != 0)] = 1

        return image, label, class_label, self.name_list[idx], frame_available, mask_available

    @staticmethod
    def numpy_to_tensor(input_):
        input_ = np.expand_dims(input_.squeeze(-1), -1).copy()
        input_ = torch.from_numpy(input_.transpose((2, 0, 1)))
        return input_
