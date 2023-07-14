import os
import torch
import torchvision
from torch.utils.data import Dataset
from dataset.avss.audio.audio_dataset import AudioDataset
from dataset.avss.visual.visual_dataset import VisualDataset
from dataset.avss.visual.visual_aug import VisualAugmentation
from typing import Callable, Dict, Optional, Tuple, Type, Union, List
from torch.utils.data._utils.collate import default_collate_fn_map
from loguru import logger
from config.avss.config_avsbench_semantics import cfg as avss_cfg
import pandas as pd

class AVSSDataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        logger.info("LOAD AVSBench-Semantics DATALOADER")
        csv_fn = avss_cfg.DATA.META_CSV_PATH
        dataframe = pd.read_csv(csv_fn, sep=',')
        dataframe = dataframe[dataframe['split'] == mode] # train, val, test
        transform_v = VisualAugmentation(
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_width=args.image_width,
            image_height=args.image_height,
            mode=mode,
            setup=args.setup
        )

        self.dataset_v = VisualDataset(
            dataframe,
            mode,
            transform=transform_v,
            args=args
        )

        self.dataset_a = AudioDataset(
            self.dataset_v.name_list,
            dataframe,
            mode,
            audio_len=10,
            transform=None,
            args=args
        )
        self.mode = mode

    def __len__(self):
        return len(self.dataset_v.name_list)

    def __getitem__(self, idx):
        if self.mode == "train":
            audio_waveform = self.dataset_a.__getitem__(idx)
            image, label, category, file_id, frame_available, mask_available = self.dataset_v.__getitem__(idx)
            return image, audio_waveform, label, category, file_id, frame_available, mask_available
        else:
            audio_waveform = self.dataset_a.__getitem__(idx)
            image, label, category, file_id, frame_available, mask_available = self.dataset_v.__getitem__(idx)
            return image, audio_waveform, label, category, file_id, frame_available, mask_available
