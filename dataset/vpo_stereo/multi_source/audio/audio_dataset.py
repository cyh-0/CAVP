import os.path

import torch
import torchaudio
from torch.utils.data import Dataset


def get_avs_fn(mode, data_frame, args=None):
    data_path = args.data_path
    if mode == "train" and args.use_synthetic:
        audio_list = [
            os.path.join(args.synth_data_path, "audios", i + ".wav")
            for i in data_frame["vgg_file"]
        ]
    # Unpair EVAL experiments
    elif mode == "val":
        tmp = data_frame[data_frame["split"] == "val"]
        gun_ = tmp[tmp["category"] == "cap_gun_shooting,background"]
        male_ = tmp[tmp["category"] == "male_speech,background"]
        male_ = male_.head(len(gun_))
        audio_list = [
            os.path.join(data_path, "audios", i + ".wav") for i in male_["name"]
        ]
    else:
        audio_list = [
            os.path.join(data_path, "audios", i + ".wav") for i in data_frame["name"]
        ]
    return audio_list


class AudioDataset(Dataset):
    def __init__(self, name_list, data_frame, mode, audio_len=0.96, transform=None, args=None):
        super().__init__()
        self.mode = mode
        self.transform_a = transform
        self.name_list = name_list
        self.audio_list = []
        self.pos_list = []
        if args.use_vpo:
            for img_id in name_list:
                tmp = data_frame[data_frame["img_Id"] == img_id]
                self.audio_list.append(tmp.audio_fp.tolist())
                self.pos_list.append(tmp.audio_pos.tolist())
        else:
            self.audio_list = get_avs_fn(mode, data_frame, args=args)

        assert len(self.audio_list) == len(self.pos_list)
        self.audio_len = audio_len
        self.args = args

    def __getitem__(self, idx, flip=False):
        fn_list = self.audio_list[idx]
        pos_list = self.pos_list[idx]
        wave_list = []
        for i in range(len(fn_list)):
            fn = fn_list[i]
            pos = pos_list[i]
            pos = 1 - pos if flip else pos
            wave_, rate_ = torchaudio.load(fn)
            wave_ = torchaudio.transforms.Resample(rate_, 16000)(wave_)
            wave_ = self.crop_audio(wave_)
            wave_ = self.add_position_info_(wave_, pos)
            wave_list.append(wave_)
        # return wave_
        return torch.stack(wave_list).sum(0)
    
    def add_position_info_(self, wave_, pos):
        L = wave_ * (1 - pos)
        R = wave_ * pos
        stereo_ = torch.cat([L, R], dim=0)
        return stereo_

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
        return len(self.audio_list)
