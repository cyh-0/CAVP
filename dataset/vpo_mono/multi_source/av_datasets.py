import os
import torch
import torchvision
from torch.utils.data import Dataset
from dataset.vpo_mono.multi_source.audio.audio_dataset import AudioDataset
from dataset.vpo_mono.multi_source.visual.visual_dataset import VisualDataset
from dataset.vpo_mono.multi_source.visual.visual_aug import VisualAugmentation
from typing import Callable, Dict, Optional, Tuple, Type, Union, List
from torch.utils.data._utils.collate import default_collate_fn_map
from loguru import logger

# def collate_list_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
#     return [torch.tensor(item) for item in batch]

# default_collate_fn_map.update({List: collate_list_fn})


class AudioVisualDataset(Dataset):
    def __init__(self, args, mode, dataframe):
        super().__init__()
        logger.info("LOAD MULTI-SOURCE DATALOADER")
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
            audio_len=args.audio_len,
            transform=None,
            args=args
        )
        self.mode = mode

    def __len__(self):
        return len(self.dataset_v.name_list)

    def __getitem__(self, idx):
        if self.mode == "train":
            audio_waveform = self.dataset_a.__getitem__(idx)
            image, label, category, file_id = self.dataset_v.__getitem__(idx)
            return image, audio_waveform, label, category, file_id
        else:
            audio_waveform = self.dataset_a.__getitem__(idx)
            image, label, category, file_id = self.dataset_v.__getitem__(idx)
            return image, audio_waveform, label, category, file_id
