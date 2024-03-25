import os
import torch
import numpy
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

# from config.class_list import index_table_avs, index_table_coco, coco_class_dict


def process_coco_fn(x, root_name, ext="jpg", mask=False, setup=None):
    c1, c2, c3 = x["cateName"], x["img_Id"], x["ann_Ids"]
    img_n = str(c2).zfill(12)
    mask_n = str(c3).zfill(12)
    if mask:
        fn = os.path.join(root_name, f"{img_n}_{mask_n}.{ext}")
    else:
        fn = os.path.join(root_name, f"{img_n}.{ext}")

    if setup == "vpo_msmi":
        if x.multi_instance == 0:
            fn = fn.replace("VPO-MSMI", "VPO-MS")
    return fn


def prepare_train_data(df, args=None):
    """
    ORG COCO
    """
    setup_ = args.setup
    if args.replace_name:
        df = df.replace({"male": "person", "female": "person", "baby": "person"})
        df["cateId"] = df["cateId"].replace({92: 1, 93: 1, 94: 1})
    ##########################
    df["audio_fp"] = df["vgg_file"].apply(
        lambda x: os.path.join(args.vgg_data_path, "audios", x + ".wav")
    )
    df["image_fp"] = df.apply(
        lambda x: process_coco_fn(x, args.coco_img_root, "jpg", setup=setup_), axis=1
    )
    df["mask_fp"] = df.apply(
        lambda x: process_coco_fn(x, args.coco_mask_root, "png", setup=setup_), axis=1
    )
    df["split"] = df["split"].replace("val", "test")
    return df


def get_avs_fn(mode, data_frame, args=None, multi_class_or_not=True):
    data_path = args.data_path
    # if mode == "val":
    #     tmp = data_frame[data_frame["split"] == "val"]
    #     data_frame = tmp[tmp["category"] == "cap_gun_shooting,background"]

    name_list = data_frame["name"].tolist()
    frame_list = [os.path.join(data_path, "frames", i + ".png") for i in name_list]
    label_list = [
        os.path.join(
            data_path, "labels" if multi_class_or_not else "salient_labels", i + ".png"
        )
        for i in name_list
    ]
    category_list = [
        [args.index_table.index(j) for j in i.split(",")]
        for i in data_frame["category"].tolist()
    ]
    return name_list, frame_list, label_list, category_list


def get_coco_fn(mode, data_frame, args=None, multi_class_or_not=True):
    data_path = args.coco_root
    # data_frame = prepare_train_data(data_frame.copy(), args)

    if mode == "val":
        data_frame = data_frame[data_frame["split"] == "val"]

    name_list = (
        data_frame["img_Id"].astype(str) + "_" + data_frame["ann_Ids"].astype(str)
    ).tolist()
    frame_list = data_frame.image_fp.tolist()
    label_list = data_frame.mask_fp.tolist()
    # Add 0 for background
    coco_cateid = data_frame.cateId.tolist()
    category_list = [
        [args.index_table.index(j) for j in i.split(",")] + [0]
        for i in data_frame["cateName"].tolist()
    ]
    return name_list, frame_list, label_list, category_list, coco_cateid


def get_coco_ms_fn(mode, data_frame, args):
    if mode == "val":
        data_frame = data_frame[data_frame["split"] == "val"]

    name_list = data_frame.img_Id.unique().tolist()
    frame_list = data_frame.image_fp.unique().tolist()
    label_list = data_frame.mask_fp.unique().tolist()
    category_list = []
    for img_id in name_list:
        tmp = data_frame[data_frame["img_Id"] == img_id]
        curr = [args.index_table.index(i) for i in tmp["cateName"].tolist()] + [0]
        category_list.append(curr)

    return name_list, frame_list, label_list, category_list


class VisualDataset(Dataset):
    def __init__(self, data_frame, mode, transform=None, args=None):
        super().__init__()
        self.mode = mode
        self.use_vpo = False
        self.num_classes = args.num_classes
        self.index_table = args.index_table
        """ COCO ORG CLASS INDEX """
        self.class_dict = args.class_dict
        self.use_vpo = True
        self.name_list, self.frame_list, self.label_list, self.category_list = get_coco_ms_fn(
            mode, data_frame, args=args
        )
        self.transform = transform

    def __getitem__(self, idx):
        # img_id = self.name_list[idx]
        fn_img = self.frame_list[idx]
        fn_label = self.label_list[idx]
        image = Image.open(fn_img).convert("RGB")
        label = Image.open(fn_label)
        image, label = self.transform(image, label)
        # TODO
        # Temporal fix to remap segmentation mask
        # if len(torch.unique(label)) > 1:
        if self.use_vpo:
            # tmp = self.category_list[idx].copy()
            # tmp.remove(0)
            tmp = torch.unique(label.clone())
            tmp = tmp[tmp != 0]
            tmp = tmp[tmp != 255].tolist()
            """ Overwrite target based on re-defined index """
            """ Remap COCO index to VPO index """
            for i_cc in tmp:
                ow_targe = self.index_table.index(self.class_dict[str(i_cc)])
                label[label == i_cc] = ow_targe

        class_label = F.one_hot(
            torch.unique(label[label != 255]), num_classes=self.num_classes
        ).sum(0)

        return image, label, class_label, self.name_list[idx]

    @staticmethod
    def numpy_to_tensor(input_):
        input_ = numpy.expand_dims(input_.squeeze(-1), -1).copy()
        input_ = torch.from_numpy(input_.transpose((2, 0, 1)))
        return input_
