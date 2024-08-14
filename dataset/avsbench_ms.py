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
from config.avsbench.config_avsbench_ms import get_cfg

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


class MS3Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', args=None):
        super().__init__()
        self.split = split
        self.mode = split
        self.num_classes = args.num_classes
        self.mask_num = 5
        self.cfg = get_cfg(args.data_root)
        df_all = pd.read_csv(self.cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        #
        self.df_train = self.df_split.copy()        
        self.df_train.loc[:, "image_path"] = self.df_train["video_id"]
        self.df_train.loc[:, "mask_path"] = self.df_train["video_id"]
        self.df_train.loc[:, "wav_path"] = self.df_train["video_id"]
        self.df_train.loc[:, "img_id"] = pd.Series()
        df_list = []
        for img_id in range(1, 6):
            tmp = self.df_train.copy()
            tmp.loc[:, "image_path"] = tmp.loc[:, "image_path"].apply(lambda x: os.path.join(self.cfg.DATA.DIR_IMG, x, "%s.mp4_%d.png"%(x, img_id))) 

            tmp.loc[:, "mask_path"] = tmp.loc[:, "mask_path"].apply(lambda x: os.path.join(self.cfg.DATA.DIR_MASK, self.split, x, "%s_%d.png"%(x, img_id)))
            tmp.loc[:, "wav_path"] = tmp.loc[:, "wav_path"].apply(lambda x: os.path.join(self.cfg.DATA.DIR_AUDIO_WAV, self.split, x + '.wav'))
            tmp.loc[:, "img_id"] = img_id
            df_list.append(tmp)
        self.df_train = pd.concat(df_list, ignore_index=True)


        # os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id))
        # os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id))
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
        if self.mode == 'train':
            self.df_train = pd.concat([self.df_train]*2)

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

        # audio_lm_path = os.path.join(self.cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, video_name + '.pkl')
        # audio_log_mel = load_audio_lm(audio_lm_path)
        
        if self.mode == 'train':
            # images = self.get_image(img_base_path, video_name)
            # labels = self.get_mask(mask_base_path, video_name)
            item = self.df_train.iloc[index]
            video_name = item["video_id"]
            images = [Image.open(item["image_path"]).convert(mode='RGB')]
            labels = [Image.open(item["mask_path"]).convert(mode='1')]
            pack_ = [self.transform(image, label) for image, label in zip(images, labels)]
            image, label = list(zip(*pack_))
            imgs_tensor = torch.stack(list(image))
            masks_tensor = torch.stack(list(label))
            audio_wav, rate_ = torchaudio.load(item["wav_path"])
            audio_wav = torchaudio.transforms.Resample(rate_, 16000)(audio_wav)
            audio_wav = self.crop_audio(audio_wav)
            audio_wav = torch.mean(audio_wav, dim=0).unsqueeze(0)
            audio_wav = audio_wav.view(1, 10, -1)
            audio_wav = audio_wav[:, item["img_id"] - 1, :]
            class_l = (masks_tensor.view(1, -1).sum(-1) != 0).long()
            class_label = F.one_hot(class_l, num_classes=2)
            frame_available = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])#.bool()
            mask_available = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            return imgs_tensor, audio_wav, masks_tensor, class_label, video_name, frame_available, mask_available
        else:
            df_one_video = self.df_split.iloc[index]
            video_name = df_one_video["video_id"]
            img_base_path =  os.path.join(self.cfg.DATA.DIR_IMG, video_name)
            mask_base_path = os.path.join(self.cfg.DATA.DIR_MASK, self.split, video_name)
            waveform_path = os.path.join(self.cfg.DATA.DIR_AUDIO_WAV, self.split, video_name + '.wav')
            imgs, masks = [], []
            for img_id in range(1, 6):
                img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s.mp4_%d.png"%(video_name, img_id)), transform=self.img_transform)
                imgs.append(img)
            for mask_id in range(1, self.mask_num + 1):
                mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='P')
                masks.append(mask)
            imgs_tensor = torch.stack(imgs, dim=0)
            masks_tensor = torch.stack(masks, dim=0)

            audio_wav, rate_ = torchaudio.load(waveform_path)
            audio_wav = torchaudio.transforms.Resample(rate_, 16000)(audio_wav)
            audio_wav = self.crop_audio(audio_wav)
            audio_wav = torch.mean(audio_wav, dim=0).unsqueeze(0)


            class_l = (masks_tensor.view(5, -1).sum(-1) != 0).long()
            class_label = F.one_hot(class_l, num_classes=2)

            frame_available = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
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
        if self.mode == 'train':
            return len(self.df_train)
        else:
            return len(self.df_split)




