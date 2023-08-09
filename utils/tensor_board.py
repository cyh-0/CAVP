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
        # os.environ["WANDB_API_KEY"] = config["wandb_key"]
        # os.system("wandb login")
        # os.system("wandb {}".format(config["wandb_mode"]))
        os.environ["WANDB_START_METHOD"] = "fork"
        run_name = "[{}][{}][BS{:02}][G{}]{}".format(
            "MS" if config.setup == "coco_ms" else "SS",
            config.visual_backbone,
            config.batch_size,
            config.gpus,
            config["experiment_name"],
        )
        self.tensor_board = wandb.init(
            # key=config["wandb_key"],
            mode=config["wandb_mode"],
            project=config["proj_name"],
            name=run_name,
            notes=config.run_note,
            tags=config.tags,
            config=config,
            settings=wandb.Settings(code_dir="."),
        )

        self.restore_transform = torchvision.transforms.Compose(
            [
                DeNormalize(config["image_mean"], config["image_std"]),
                torchvision.transforms.ToPILImage(),
            ]
        )
        self.pallete = get_pallete(config.num_classes)

    def upload_wandb_image(self, image, gt, pred, score, heatmap=None, status="", folder=""):
        # image
        denorm_image = [
            numpy.asarray(self.restore_transform(i.squeeze()), dtype=numpy.int)
            for i in image
        ]

        # y_tilde
        pred = torch.argmax(pred, dim=1).long()

        # attention
        score = torch.sum(score[:, 1:], dim=1).numpy()
        score = numpy.uint8((1 - score.copy()) * 255)

        # pseudo-label lists
        pred[gt == 255] = 255
        pred_lists = [colorize_mask(i, self.pallete) for i in pred.numpy()]

        # pixel-wise label lists
        gt = gt.numpy()
        gt_lists = [colorize_mask(i, self.pallete) for i in gt]

        score[gt == 255] = 255
        score_lists = [
            cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in score
        ]
        score_lists = [
            cv2.addWeighted(score_lists[i], 0.7, numpy.uint8(denorm_image[i]), 0.3, 0)
            for i in range(len(score_lists))
        ]
        if heatmap is not None:
            for i in range(heatmap.shape[0]):
                curr_denorm_image = denorm_image[i]
                curr_gt = numpy.repeat(gt[i,None], 4, axis=0)
                h_map = interpolate(heatmap[i], size=gt.shape[-2:], mode="bilinear", align_corners=False)
                h_map = h_map.squeeze().numpy()
                h_map = numpy.uint8((h_map.copy()) * 255)
                h_map[curr_gt == 255] = 255
                h_map_lists = [
                    cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in h_map
                ]
                h_map_lists = [
                    cv2.addWeighted(h_map_lists[i], 0.7, numpy.uint8(curr_denorm_image), 0.3, 0)
                    for i in range(len(h_map_lists))
                ]
                wandb.log({"h_{}".format(status): [wandb.Image(i) for i in h_map_lists]})

        wandb.log({"x_{}".format(status): [wandb.Image(i) for i in denorm_image]})
        wandb.log({"y_{}".format(status): [wandb.Image(i) for i in gt_lists]})
        wandb.log({"y_tilde_{}".format(status): [wandb.Image(i) for i in pred_lists]})
        # wandb.log({"{}/s_{}".format(folder,status): [wandb.Image(i) for i in score_lists]})

    def upload_wandb_heatmap(self, denorm_image, gt, heatmap, status="", folder=""):
        for i in range(heatmap.shape[0]):
            score = heatmap[i].squeeze().numpy()
            score = numpy.uint8((1 - score.copy()) * 255)
            score[gt == 255] = 255
            score_lists = [
                cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in score
            ]
            score_lists = [
                cv2.addWeighted(score_lists[i], 0.7, numpy.uint8(denorm_image[i]), 0.3, 0)
                for i in range(len(score_lists))
            ]
            wandb.log({"x_{}".format(status): [wandb.Image(i) for i in denorm_image]})


    def upload_wandb_image_backup(
        self, waveform, image, gt, pred, score, status="", mode="eval"
    ):
        # image
        denorm_image = [
            numpy.asarray(self.restore_transform(i.squeeze()), dtype=numpy.int)
            for i in image
        ]

        # y_tilde
        pred = torch.argmax(pred, dim=1).long()

        # attention
        score = torch.mean(score, dim=1).numpy()
        score = numpy.uint8((1 - score.copy()) * 255)

        # pseudo-label lists
        pred[gt == 255] = 255
        pred_lists = [colorize_mask(i, self.pallete) for i in pred.numpy()]

        # pixel-wise label lists
        gt = gt.numpy()
        gt_lists = [colorize_mask(i, self.pallete) for i in gt]

        score[gt == 255] = 255
        score_lists = [
            cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in score
        ]
        score_lists = [
            cv2.addWeighted(score_lists[i], 0.7, numpy.uint8(denorm_image[i]), 0.3, 0)
            for i in range(len(score_lists))
        ]

        # tmp = numpy.unique(gt)
        # tmp[tmp!=0]
        # print_waveform(waveform, "111")
        canvas = numpy.zeros_like(denorm_image[0])
        label = numpy.asarray(gt_lists[0].convert("RGB"))

        canvas[label != 0] = label[label != 0]
        canvas[label != 0] = denorm_image[0][label != 0]

        result = [(denorm_image[0].astype(float) * .3 + label.astype(float) * .7).astype(numpy.uint8)]
        

        wandb.log({"{}_x".format(status): [wandb.Image(i) for i in result]})
        # wandb.log({"{}_x".format(status): [wandb.Image(i) for i in denorm_image]})
        # wandb.log({"{}_y".format(status): [wandb.Image(i) for i in gt_lists]})
        # wandb.log({"{}_y_tilde".format(status): [wandb.Image(i) for i in pred_lists]})
        wandb.log({"{}_s".format(status): [wandb.Image(i) for i in score_lists]})

    def upload_metrics(self, info_dict, epoch=None):
        for i, info in enumerate(info_dict):
            tmp_dict = {info: info_dict[info]}
            tmp_dict.update({"epoch": epoch}) if epoch is not None else None
            self.tensor_board.log(tmp_dict)
        return
    
    def upload_ood_image(self, energy_map, img_number=4, data_name="?"):
        self.tensor_board.log(
            {
                "{}_focus_area_map".format(data_name): [
                    wandb.Image(j, caption="id {}".format(str(i)))
                    for i, j in enumerate(energy_map[:img_number])
                ]
            }
        )
        return

    def generate_image(self, image, gt, pred, score, status=""):
        # image
        denorm_image = [
            numpy.asarray(self.restore_transform(i.squeeze()), dtype=numpy.int)
            for i in image
        ]
        pred = torch.argmax(pred, dim=1).long()

        # attention
        score = torch.sum(score[:, 1:], dim=1).numpy()
        score = numpy.uint8((1 - score.copy()) * 255)

        # pseudo-label lists
        pred[gt == 255] = 255
        pred_lists = [colorize_mask(i, self.pallete) for i in pred.numpy()]

        # pixel-wise label lists
        gt = gt.numpy()
        gt_lists = [colorize_mask(i, self.pallete) for i in gt]

        score[gt == 255] = 255
        score_lists = [
            cv2.applyColorMap(i[..., numpy.newaxis], cv2.COLORMAP_JET) for i in score
        ]
        score_lists = [
            cv2.addWeighted(score_lists[i], 0.7, numpy.uint8(denorm_image[i]), 0.3, 0)
            for i in range(len(score_lists))
        ]
        return denorm_image, gt_lists, pred_lists, score_lists

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
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid