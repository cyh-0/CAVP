import os
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import wandb
from PIL import Image
from torch.nn.functional import interpolate
from loguru import logger

# fmt: off

def print_waveform(waveform, name):
    color = [255, 255, 255]
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 1)

    ax.set_facecolor(numpy.array(color) / 255.0)
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        # labelbottom=False
    )  # labels along the bottom edge are off

    plt.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        length=0,
    )  # labels along the bottom edge are off
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    wav = (
        waveform[0]
        .cpu()
        .detach()
        .view(
            -1,
        )
        .numpy()
    )

    plt.xlim(0, len(wav))
    plt.plot(wav, color="black")
    # plt.savefig(os.path.join(wandb.run.dir,,"{}_audio.png").format(name), bbox_inches="tight", pad_inches=0.03)
    fig = plt.gcf()
    return fig
    # return Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())


class Tensorboard:
    def __init__(self, config):
        logger.critical(f"SETTING WANDB DIR TO >>> {config.wandb_dir}")
        os.environ["WANDB_START_METHOD"] = "fork"
        os.environ["WANDB_API_KEY"] = config["wandb_key"]
        run_name = "[{}][{}][BS{:02}][G{}]{}".format(
            config.setup.upper(),
            config.visual_backbone,
            config.batch_size,
            config.gpus,
            config["experiment_name"],
        )
        self.tensor_board = wandb.init(
            mode=config["wandb_mode"],
            project=config["proj_name"],
            name=run_name,
            notes=config.run_note,
            tags=config.tags,
            config=config,
            dir=config.wandb_dir,
            settings=wandb.Settings(code_dir="."),
        )

        self.restore_transform = torchvision.transforms.Compose(
            [
                DeNormalize(config["image_mean"], config["image_std"]),
                torchvision.transforms.ToPILImage(),
            ]
        )

        self.pallete = get_pallete(config.num_classes)
        self.img_size_vis = (256, 256)

    def upload_wandb_image(self, image, gt, pred, score, heatmap=None, 
                           status="", folder="", caption=None, resize=False, 
                           show_x=True, show_y=True):
        # image
        denorm_image = [
            numpy.asarray(self.restore_transform(i.squeeze()), dtype=int) for i in image
        ]

        # y_tilde
        pred = torch.argmax(pred, dim=1).long()

        # pseudo-label lists
        pred[gt == 255] = 255
        pred_lists = [colorize_mask(i, self.pallete) for i in pred.numpy()]

        # pixel-wise label lists
        gt = gt.numpy()
        gt_lists = [colorize_mask(i, self.pallete) for i in gt]

        h_map_lists = []
        if heatmap is not None:
            for i in range(heatmap.shape[0]):
                curr_denorm_image = denorm_image[i]
                h_map = torchvision.utils.make_grid(heatmap[i].unsqueeze(1), nrow=10)[0]
                h_map = h_map.squeeze().numpy()
                h_map = numpy.uint8((h_map.copy()) * 255)
                h_map_lists.append(Image.fromarray(h_map))
                # h_map_lists = [cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in h_map]
                # h_map_lists = [cv2.addWeighted(h_map_lists[i], 0.7, numpy.uint8(curr_denorm_image), 0.3, 0) for i in range(len(h_map_lists))]
                # wandb.log({os.path.join(folder, f"h_{status}"): [wandb.Image(i) for i in h_map_lists]})

        for i in range(len(denorm_image)):
            caption_ = caption[i] if caption is not None else None
            img_ = Image.fromarray(denorm_image[i].astype(numpy.uint8))
            gt_ = gt_lists[i]
            pred_ = pred_lists[i]
            heatmap_ = h_map_lists[i] if heatmap is not None else None
            if resize:
                img_ = img_.resize(self.img_size_vis, Image.NEAREST)
                gt_ = gt_.resize(self.img_size_vis, Image.NEAREST)
                pred_ = pred_.resize(self.img_size_vis, Image.NEAREST)

            wandb.log(
                {
                    os.path.join(folder, f"x_{status}"): wandb.Image(img_, caption=caption_) if show_x else None ,
                    os.path.join(folder, f"y_{status}"): wandb.Image(gt_, caption=caption_) if show_y else None ,
                    os.path.join(folder, f"y_tilde_{status}"): wandb.Image(pred_, caption=caption_),
                    os.path.join(folder, f"h_{status}"): wandb.Image(heatmap_, caption=caption_) if heatmap_ is not None else None,
                }
            )

    def log_image(self, image_list, prefix=None, name="sample_image"):
        section = f"{prefix}/{name}" if prefix is not None else name
        self.tensor_board.log({section: [item for item in image_list]})
        return

    def upload_wandb_heatmap(self, denorm_image, gt, heatmap, status="", folder=""):
        for i in range(heatmap.shape[0]):
            score = heatmap[i].squeeze().numpy()
            score = numpy.uint8((1 - score.copy()) * 255)
            score[gt == 255] = 255
            score_lists = [
                cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET)
                for i in score
            ]
            score_lists = [
                cv2.addWeighted(
                    score_lists[i], 0.7, numpy.uint8(denorm_image[i]), 0.3, 0
                )
                for i in range(len(score_lists))
            ]
            wandb.log({"x_{}".format(status): [wandb.Image(i) for i in denorm_image]})

    def upload_metrics(self, info_dict, epoch=None):
        for i, info in enumerate(info_dict):
            tmp_dict = {info: info_dict[info]}
            tmp_dict.update({"epoch": epoch}) if epoch is not None else None
            self.tensor_board.log(tmp_dict)
        return

    def generate_pred(self, pred, status=""):
        # pixel-wise label lists
        pred = torch.argmax(pred, dim=1).long()
        # pseudo-label lists
        pred_lists = [colorize_mask(i, self.pallete) for i in pred.numpy()]
        return pred_lists

    def generate_image(self, image=None, gt=None, pred=None):
        output = {}
        # image
        if image is not None:
            image_ = image.clone().cpu().detach()
            denorm_image = [
                numpy.asarray(self.restore_transform(i.squeeze()), dtype=int)
                for i in image_
            ]
            denorm_image = [
                Image.fromarray(item.astype(numpy.uint8)) for item in denorm_image
            ]
            output.update({"image": denorm_image})

        # pixel-wise label lists
        if gt is not None:
            gt_ = gt.clone().cpu().detach()
            gt_lists = [colorize_mask(i, self.pallete) for i in gt_.numpy()]
            output.update({"gt": gt_lists})

        if pred is not None:
            pred_ = pred.clone().cpu().detach()
            pred_ = torch.argmax(pred_, dim=1).long()
            # pseudo-label lists
            # pred_[gt_ == 255] = 255
            pred_lists = [colorize_mask(i, self.pallete) for i in pred_.numpy()]
            output.update({"pred": pred_lists})
        return output

    @staticmethod
    def finish():
        wandb.finish()


def get_pallete(num_classes):
    if num_classes == 2:
        return [0, 0, 0, 255, 255, 255]
    pallete = [0] * (num_classes * 3)
    for j in range(0, num_classes):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    # the ignore mask;
    palette[-3:] = [255, 255, 255]

    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert("P")
    new_mask.putpalette(palette)
    return new_mask


def pil_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
