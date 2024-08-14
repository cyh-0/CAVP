import os
from wave import _wave_params
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
from easydict import EasyDict as edict
from config.class_list import index_table_avs
import torch.nn.functional as F
# from config import cfg
import pdb
from dataset.avss.visual.visual_aug import VisualAugmentation
from config.avsbench.config_avsbench_s4 import get_cfg


def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', args=None):
        super().__init__()
        self.split = split
        self.mode = split
        self.num_classes = args.num_classes
        self.mask_num = 1 if self.split == 'train' else 5
        self.cfg = get_cfg(args.data_root)
        df_all = pd.read_csv(self.cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        self.audio_len = 10
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform = VisualAugmentation(
            image_mean=args.image_mean,
            image_std=args.image_std,
            image_width=args.image_width,
            image_height=args.image_height,
            mode=self.mode,
            setup=args.setup
        )
        self.index_table = index_table_avs

    def get_image(self, img_base_path, video_name):
        imgs = []
        for img_id in range(1, 6):
            path = os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id))
            img = Image.open(path).convert(mode='RGB')
            imgs.append(img)
        return imgs
    
    def get_mask(self, mask_base_path, video_name):
        mask_num = self.mask_num
        labels = []
        if self.mode != 'train':
            assert mask_num == 5
        total_mask_num = 5
        label_pad_zero_num = 5 - mask_num
        for mask_id in range(1, mask_num + 1):
            mask_path = os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id))
            mask_ = Image.open(mask_path).convert(mode='1')
            labels.append(mask_)
        for pad_j in range(label_pad_zero_num):
            mask_ = Image.fromarray(np.zeros_like(np.asarray(mask_)))
            labels.append(mask_)
        return labels

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(self.cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_lm_path = os.path.join(self.cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(self.cfg.DATA.DIR_MASK, self.split, category, video_name)
        waveform_path = os.path.join(self.cfg.DATA.DIR_AUDIO_WAV, self.split, category, video_name + '.wav')
        audio_log_mel = load_audio_lm(audio_lm_path)
        
        if self.mode == 'train':
            images = self.get_image(img_base_path, video_name)
            labels = self.get_mask(mask_base_path, video_name)
            pack_ = [self.transform(image, label) for image, label in zip(images, labels)]
            image, label = list(zip(*pack_))
            imgs_tensor = torch.stack(list(image))
            masks_tensor = torch.stack(list(label))
        else:
            imgs, masks = [], []
            for img_id in range(1, 6):
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
                masks.append(mask)
            imgs_tensor = torch.stack(imgs, dim=0)
            masks_tensor = torch.stack(masks, dim=0)


        audio_wav, rate_ = torchaudio.load(waveform_path)
        audio_wav = torchaudio.transforms.Resample(rate_, 16000)(audio_wav)
        audio_wav = self.crop_audio(audio_wav)
        audio_wav = torch.mean(audio_wav, dim=0).unsqueeze(0)

        if self.num_classes <= 2:
            class_label = torch.tensor([0, 1]).view(1,-1)
        else:
            class_label = F.one_hot(
                    torch.tensor([self.index_table.index(category)]), num_classes=len(self.index_table)
                )
        frame_available = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
        if self.split == 'train':
            mask_available = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            mask_available = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        return imgs_tensor, audio_wav, masks_tensor, class_label, video_name, frame_available, mask_available

    def crop_audio(self, audio_waveform, sr=16000):
        mid_ = audio_waveform.shape[1] // 2
        sample_len = int(self.audio_len * sr)
        st = mid_ - sample_len // 2
        et = st + sample_len
        audio_waveform = audio_waveform[:, st:et]

        if audio_waveform.shape[1] != sample_len:
            r_num = sample_len//audio_waveform.shape[1] + 1
            audio_waveform = audio_waveform.repeat(1, r_num)[:, :sample_len]

        return audio_waveform

    def __len__(self):
        return len(self.df_split)




if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
