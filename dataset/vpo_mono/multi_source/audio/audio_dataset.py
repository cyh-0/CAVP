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
        if args.use_vpo:
            for img_id in name_list:
                tmp = data_frame[data_frame["img_Id"] == img_id]
                self.audio_list.append(tmp.audio_fp.tolist())
        else:
            self.audio_list = get_avs_fn(mode, data_frame, args=args)

        self.audio_len = audio_len
        self.args = args

    def __getitem__(self, idx):
        fn_list = self.audio_list[idx]
        wave_list = []
        for fn in fn_list:
            wave_, rate_ = torchaudio.load(fn)
            wave_ = torchaudio.transforms.Resample(rate_, 16000)(wave_)
            wave_ = self.crop_audio(wave_)
            wave_list.append(wave_)
        # return wave_
        return torch.stack(wave_list).sum(0)

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
