from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from scipy import signal
import json
import xml.etree.ElementTree as ET


def inverse_normalize(tensor):
    inverse_mean = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    inverse_std = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor


def convert_normalize(tensor, new_mean, new_std):
    raw_mean = IMAGENET_DEFAULT_MEAN
    raw_std = IMAGENET_DEFAULT_STD
    # inverse_normalize with raw mean & raw std
    inverse_mean = [-mean / std for mean, std in zip(raw_mean, raw_std)]
    inverse_std = [1.0 / std for std in raw_std]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    # normalize with new mean & new std
    tensor = transforms.Normalize(new_mean, new_std)(tensor)
    return tensor


def load_image(path, mode="RGB"):
    return Image.open(path).convert(mode)


# def load_audio_feature(args, file_id):
#     obj = np.load(os.path.join(args.train_data_path, "audio_feature", f"{file_id}.npy",))
#     return obj.copy()


def load_all_bboxes(annotation_dir, format="flickr"):
    gt_bboxes = {}
    if format == "flickr":
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split(".")[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == "bbox":
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text) / 256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == "vggss":
        with open("metadata/vggss.json") as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [
                (np.clip(np.array(bbox), 0, 1) * 224).astype(int)
                for bbox in annotation["bbox"]
            ]
            gt_bboxes[annotation["file"]] = bboxes
    else:
        return None

    return gt_bboxes


def bbox2gtmap(bboxes, format="flickr"):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == "flickr":
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format == "vggss":
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map
