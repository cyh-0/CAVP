import random
import numpy
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from loguru import logger

class VisualAugmentation(object):
    def __init__(self, image_mean, image_std, image_width, image_height, mode, setup):
        self.mode = mode
        self.image_size = (image_height, image_width)
        self.image_norm = (image_mean, image_std)
        self.color_jitter = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.25)
        self.normalise = transforms.Normalize(mean=image_mean, std=image_std)
        self.get_crop_pos = transforms.RandomCrop(self.image_size)
        self.to_tensor = transforms.ToTensor()
        if setup == "avs":
            # AVS
            self.scale_list = [.5, .75, 1.]
            self.color_jitter = None
        else:
            # COCO
            # self.scale_list = [.75, 1., 1.25, 1.5, 1.75, 2.]
            self.scale_list = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]
        
        logger.critical(f"ColorJitter: {self.color_jitter}")
        logger.critical(f"Scale list: {self.scale_list}")

    def random_crop_with_padding(self, image_, label_):
        w_, h_ = image_.size
        if min(h_, w_) < min(self.image_size):
            res_w_ = max(self.image_size[0] - w_, 0)
            res_h_ = max(self.image_size[1] - h_, 0)
            image_ = F.pad(image_, [0, 0, res_w_, res_h_], fill=(numpy.array(self.image_norm[0]) * 255.).tolist())
            label_ = F.pad(label_, [0, 0, res_w_, res_h_], fill=255)

        pos_ = self.get_crop_pos.get_params(image_, self.image_size)
        image_ = F.crop(image_, *pos_)
        label_ = F.crop(label_, *pos_)
        return image_, label_

    # @staticmethod
    def random_scales(self, image_, label_):
        w_, h_ = image_.size
        chosen_scale = random.choice(self.scale_list)
        w_, h_ = int(w_ * chosen_scale), int(h_ * chosen_scale)
        image_ = F.resize(image_, (h_, w_), transforms.InterpolationMode.BICUBIC)
        label_ = F.resize(label_, (h_, w_), transforms.InterpolationMode.NEAREST)
        return image_, label_

    @staticmethod
    def random_flip_h(image_, label_):
        chosen_flip = random.random() > 0.5
        image_ = F.hflip(image_) if chosen_flip else image_
        label_ = F.hflip(label_) if chosen_flip else label_
        return image_, label_, chosen_flip

    def train_aug(self, x, y):
        x, y, chosen_flip = self.random_flip_h(x, y)
        x, y = self.random_scales(x, y)
        if self.color_jitter is not None:
            x = self.color_jitter(x) 
        x, y = self.random_crop_with_padding(x, y)
        x = self.to_tensor(x)
        y = torch.tensor(numpy.asarray(y)).long()
        x = self.normalise(x)
        return x, y, chosen_flip

    def test_aug(self, x, y):
        x = self.to_tensor(x)
        y = torch.tensor(numpy.asarray(y)).long()
        x = self.normalise(x)
        return x, y

    def __call__(self, x, y):
        return self.train_aug(x, y) if self.mode == "train" else self.test_aug(x, y)



