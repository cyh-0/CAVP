from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchaudio
from tqdm import tqdm

from engine.utils import DeNormalize
from utils import sourcesep
from loguru import logger
from loss.losser import Losser
import wandb
import os
from loss.contrastive_aud import ContrastLoss
from einops import rearrange
from utils.eval_utils import MIoU
from utils.eval_utils import ForegroundDetect
from visualisation.tsne import tsne_plotter
from utils import ddp_utils
from models.mm_fusion import SoundBank
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy
from PIL import Image

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N

    return iou

class BASELINE:
    def __init__(self, args, train_loader, visual_tool, engine, lr_scheduler) -> None:
        self.args = args
        self.engine = engine
        self.vis_tool = visual_tool
        self.local_rank = self.engine.local_rank
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        self.train_loader = train_loader
        self.denorm = DeNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.lr_scheduler = lr_scheduler
        self.best_iou = 0
        self.stft = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                n_mels=64,
                f_min=125.0,
                f_max=3800.0,
            )
        ).cuda(self.local_rank)
        self.loss_func = Losser(args.num_classes, self.local_rank)
        self.loss_ctr_av = ContrastLoss(temperature=0.1, ignore_idx=255, max_views=512)
        self.ow_rate = args.ow_rate

    def preprocess_audio(self, audio):
        N, C, A = audio.size()
        n_len = 96 if self.args.audio_len == 1.0 else 300
        audio = audio.view(N * C, A)
        audio = self.stft(audio)[:, :, :n_len]  # [:, :, :96]
        audio = audio.transpose(-1, -2)
        audio = sourcesep.db_from_amp(audio, cuda=True)
        audio = sourcesep.normalize_spec(audio, self.args)
        _, T, F = audio.size()
        audio = audio.view(N, C, T, F)
        return audio

    def lr_step(self, optimizer_v_, optimizer_a_, current_lr, use_baseline=False):
        # Backbone + Cross Attn
        for _, opt_group in enumerate(optimizer_v_.param_groups[:4]):
            opt_group["lr"] = current_lr
        # Segmentation
        for _, opt_group in enumerate(optimizer_v_.param_groups[4:]):
            opt_group["lr"] = current_lr * 10.0
        # Audio
        # for _, opt_group in enumerate(optimizer_a_.param_groups):
        #     opt_group["lr"] = current_lr

        lr_bkb = optimizer_v_.param_groups[0]["lr"]
        lr_attn = optimizer_v_.param_groups[2]["lr"]
        lr_seg = optimizer_v_.param_groups[4]["lr"]
        lr_audio = self.args.lr
        # lr_audio = optimizer_a_.param_groups[0]["lr"]
        return lr_bkb, lr_attn, lr_seg, lr_audio

    def overwrite_miss_match(self, if_match, shuffle_img_label, img_label):
        """Find single source"""
        rm_background = img_label.clone()
        rm_background[:, 0] = 0
        source_count = rm_background.sum(1)
        ms_idx = torch.where(source_count != 1)[0]

        """ Config change_dix """
        false_list = (if_match == 0).nonzero(as_tuple=True)[0]
        n_false = false_list.shape[0]
        shuffle_idx = torch.randperm(n_false)[: int(n_false * self.ow_rate)]
        change_idx = false_list[shuffle_idx]

        """ Filter MS"""
        match_idx = torch.isin(change_idx, ms_idx)
        change_idx = change_idx[~match_idx]

        """ Incase we drop idx under single source setup """
        if self.args.setup != "coco_ms":
            assert match_idx.sum().item() == 0

        """ change_idx: target class label"""
        mod_idx_map = {}

        for idx in change_idx:
            idx = idx.item()
            if_match[idx] = True
            true_label = img_label[idx]
            shuffle_img_label[idx] = true_label
            # curr_label = true_label[true_label != 0].detach().cpu().item()
            tmp = true_label.nonzero().squeeze()
            curr_label = tmp[tmp.nonzero().squeeze()]
            """ Change audio """
            # fake_audio = self.sound_bank[None, curr_label][
            #     :, tmp_audio_idx[curr_label], None
            # ]
            # tmp_audio_idx[curr_label] += 1
            # shuffle_audio[idx] = self.preprocess_audio(fake_audio)[0]
            mod_idx_map.update({idx: curr_label.item()})
        return if_match, shuffle_img_label, mod_idx_map  # shuffle_audio

    @abstractmethod
    def train(
        self, model_v_, model_a_, optimizer_v_, optimizer_a_, epoch, train_loader
    ):
        model_v_.train()
        model_a_.train()
        loader_len = len(train_loader)

        if self.local_rank <= 0:
            tbar = tqdm(
                train_loader, total=loader_len, ncols=120, position=0, leave=True
            )
        else:
            tbar = train_loader

        ow_flag = True if epoch >= 1 else False

        for batch_idx, (image, waveform, pix_label, img_label, _) in enumerate(tbar):

            self.engine.update_iteration(epoch, epoch * loader_len + batch_idx)
            waveform = waveform.cuda(self.local_rank, non_blocking=True)
            image = image.cuda(self.local_rank, non_blocking=True)
            pix_label = pix_label.cuda(self.local_rank, non_blocking=True)
            img_label = img_label.cuda(self.local_rank, non_blocking=True)

            optimizer_v_.zero_grad()
            optimizer_a_.zero_grad()

            audio = self.preprocess_audio(waveform)

            # #####################################
            # embeds_match = rearrange(ctr_feature, "b (h w) c -> b c h w", h=128, w=128)
            # shuffle the audio and the mask.
            shuffle_idx = torch.randperm(image.shape[0]).cuda(self.local_rank)
            # audio_feature = audio_feature[shuffle_idx]
            # img_label = torch.stack(img_label).cuda(self.local_rank)
            shuffle_img_label = img_label.clone()[shuffle_idx]
            shuffle_pix_label = pix_label.clone()[shuffle_idx]
            if_match = torch.all(torch.eq(img_label, shuffle_img_label), dim=1)
            """"1"""
            # shuffle_audio = audio[shuffle_idx]
            mod_idx_map = None
            if epoch >= 1:
                if_match, shuffle_img_label, mod_idx_map = self.overwrite_miss_match(
                    if_match, shuffle_img_label, img_label
                )

            """"1"""
            # #####################################
            # image_in = torch.cat((image, image),dim=0)
            # audio_in = torch.cat((audio, shuffle_audio), dim=0)
            shuffle_info = {
                "shuffle_idx": shuffle_idx,
                "mod_idx_map": mod_idx_map,
                "image_label": img_label.clone().cpu().detach(),
            }
            output_cat, ctr_feature_cat = model_v_(image, audio, shuffle_info, ow_flag)
            B = image.shape[0]
            output = output_cat[:B] + output_cat[B:] * 0.0
            ctr_feature = ctr_feature_cat[:B]
            ctr_feature_shuff = ctr_feature_cat[B:]
            assert ctr_feature.shape[0] == ctr_feature_shuff.shape[0]

            #############################
            # if the shuffle audio don't match the original sources, we give background.
            shuffle_pix_label[~if_match] = 0
            # if the shuffle audio match the original sources, we give its gt.
            shuffle_pix_label[if_match] = pix_label[if_match]
            #############################

            l_ctr_av = self.loss_ctr_av(
                ctr_feature, pix_label, ctr_feature_shuff, shuffle_pix_label
            )

            l_ce, l_ctr_global, l_ctr_pixel, loss_aud_cls, l_mctr_av = self.loss_func(
                output, pix_label
            )

            loss = l_ce + l_ctr_global + l_ctr_pixel + l_ctr_av + l_mctr_av

            loss.backward()
            optimizer_v_.step()
            optimizer_a_.step()

            del (
                shuffle_idx,
                shuffle_pix_label,
                shuffle_img_label,
                ctr_feature_shuff,
                ctr_feature,
            )
            del img_label, waveform, audio

            current_lr = self.lr_scheduler.get_lr(
                cur_iter=batch_idx + loader_len * epoch
            )

            lr_bkb, lr_attn, lr_seg, lr_audio = self.lr_step(
                optimizer_v_,
                optimizer_a_,
                current_lr,
                use_baseline=self.args.use_baseline,
            )

            if self.local_rank <= 0:
                tbar.set_description(
                    "epoch {}:  l_ce {:.3f}  l_ctr_pixel {:.3f}  l_mctr_av {:.3f}  l_ctr_av {:.3f}".format(
                        epoch,
                        l_ce.item(),
                        l_ctr_pixel.item(),
                        l_mctr_av.item(),
                        l_ctr_av.item(),
                    )
                )
                if batch_idx % self.args.display_iter == 0:
                    self.vis_tool.upload_metrics(
                        {
                            "loss/loss": loss,
                            "loss/cross_entropy": l_ce.item(),
                            "loss/l_ctr_global": l_ctr_global.item(),
                            "loss/l_ctr_pixel": l_ctr_pixel.item(),
                            "loss/l_mctr_av": l_mctr_av.item(),
                            "loss/l_ctr_av": l_ctr_av.item(),
                            "loss/loss_aud_cls": loss_aud_cls.item(),
                            "lr/lr_bkb": lr_bkb,
                            "lr/lr_attn": lr_attn,
                            "lr/lr_seg": lr_seg,
                            "lr/lr_audio": lr_audio,
                        }
                    )
                if batch_idx % self.args.upload_iter == 0:
                    self.vis_tool.upload_wandb_image(
                        image.cpu().detach(),
                        pix_label.cpu().detach(),
                        output.cpu().detach(),
                        torch.softmax(output, dim=1).cpu().detach(),
                        status="train",
                    )
            del (
                image,
                output,
                pix_label,
                loss,
                l_ce,
                l_ctr_global,
                l_ctr_pixel,
                l_ctr_av,
                l_mctr_av,
            )
        return

    @torch.no_grad()
    def validation(self, model_v_, model_a_, epoch, test_loader, status="val"):
        model_v_.eval()
        model_a_.eval()
        target_list, audio_fea_list, c_list = [], [], []
        miou_measure = MIoU(
            num_classes=self.args.num_classes,
            ignore_index=255,
            local_rank=self.local_rank,
        )
        fg_measure = ForegroundDetect(num_classes=24, local_rank=self.local_rank)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        # from math import ceil

        for batch_idx, (image, waveform, pix_label, img_label, name) in enumerate(tbar):

            waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)
            index_table = test_loader.dataset.dataset_v.index_table
            """
            Get Pred
            """
            audio = self.preprocess_audio(waveform)
            if self.args.use_baseline:
                avl_map_logits = model_v_.module(image)
            else:
                (
                    avl_map_logits,
                    _,
                ) = model_v_.module(image, audio, eval_mode=True)
            """
            Prepare T-SNE
            """
            # fea = model_v_.module.audio_backbone(audio)
            # fea = model_v_.module.audio_proj(fea)
            # tar = img_label[0]
            # tar = tar[tar != 0]
            # target_list.append(tar)
            # audio_fea_list.append(fea)
            # c_list.append(index_table[tar.item()])
            """
            Prepare Metrics
            """
            fg_measure(avl_map_logits, pix_label)
            miou, acc = miou_measure(avl_map_logits, pix_label)
            tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))

        final_pix_metrics = miou_measure.get_metric_results()
        final_detect_metrics = fg_measure.get_metric_results()

        v_miou = final_pix_metrics[0]
        v_acc = final_pix_metrics[1]
        v_fd = final_detect_metrics[0]
        v_f1 = final_detect_metrics[1]

        if self.local_rank <= 0:
            logger.success(
                f"| mIoU: {v_miou:.4f} | acc: {v_acc:.4f} | fdr: {v_fd:.4f} | f1: {v_f1:.4f}"
            )
            # fea_list = torch.cat(audio_fea_list, dim=0)
            # target_list = torch.stack(target_list)
            # tsne_plotter(fea_list, target_list, c_list, num_classes=(self.args.num_classes-1), name="audio_fea")
            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f1"] = v_f1
                self.best_iou = v_miou
                self.engine.save_checkpoint(
                    os.path.join(wandb.run.dir, "best_model.pth")
                )

            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f1": v_f1},
                epoch
            )

        return

    @torch.no_grad()
    def eval(self, model_v_, model_a_, epoch, test_loader, status="val"):
        model_v_.eval()
        model_a_.eval()
        target_list, audio_fea_list, c_list = [], [], []

        miou_measure = MIoU(
            num_classes=self.args.num_classes,
            ignore_index=255,
            local_rank=self.local_rank,
        )
        fg_measure = ForegroundDetect(num_classes=24, local_rank=self.local_rank)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        # index_table = test_loader.dataset.dataset_v.index_table

        from torchmetrics import JaccardIndex
        jaccard = JaccardIndex(task="multiclass", num_classes=self.args.num_classes,average="none").cuda()

        counter = torch.zeros((self.args.num_classes,))

        for batch_idx, (image, waveform, pix_label, img_label, name) in enumerate(tbar):
            waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)
            index_table = test_loader.dataset.dataset_v.index_table
            """
            Get Pred
            """
            # waveform = torch.zeros_like(waveform).cuda(non_blocking=True)
            audio = self.preprocess_audio(waveform)

            avl_map_logits, _ = model_v_.module(image, audio, eval_mode=True)

            # fg_measure(avl_map_logits, pix_label)
            # miou, acc = miou_measure(avl_map_logits, pix_label)
            # tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))


            """ UNPAIRED IMAGE TESTING """
            # (
            #     denorm_image,
            #     gt_lists,
            #     pred_lists,
            #     score_lists,
            # ) = self.vis_tool.generate_image(
            #     image.cpu().detach(),
            #     pix_label.cpu().detach(),
            #     avl_map_logits.cpu().detach(),
            #     torch.softmax(avl_map_logits, dim=1).cpu().detach(),
            #     status=status,
            # )
            # path = "/home/yuanhong/Documents/audio_visual_project/ICCV2023/yy_audio/ckpts/save_models/images"
            # pred_lists[0].save(os.path.join(path, f"pred_{name}.png"))
            # Image.fromarray(score_lists[0]).save(
            #     os.path.join(path, f"score_{name}.png")
            # )
            # a = 120

            # if img_label.nonzero().shape[0] <= 2:
            #     continue

            # miou_list = jaccard(avl_map_logits, pix_label)
            # img_label[0][0] = 0

            # label = img_label.nonzero().view(-1)[1]
            # if counter[label] > 20:
            #     continue
            # else:
            #     counter += img_label[0]

            
            # if miou_list[1:].max() > 0.4:
            print(f"SAVING {name[0]}")
            """ VISUALISATION """
            self.vis_tool.upload_wandb_image(
                waveform,
                image.cpu().detach(),
                pix_label.cpu().detach(),
                avl_map_logits.cpu().detach(),
                torch.softmax(avl_map_logits, dim=1).cpu().detach(),
                status=name[0],
            )

        # fea_list = torch.cat(audio_fea_list, dim=0)
        # target_list = torch.stack(target_list)

        # tsne_plotter(fea_list, target_list, c_list, name="audio_fea")

        return
        final_pix_metrics = miou_measure.get_metric_results()
        final_detect_metrics = fg_measure.get_metric_results()

        v_miou = final_pix_metrics[0]
        v_acc = final_pix_metrics[1]
        v_fd = final_detect_metrics[0]
        v_f1 = final_detect_metrics[1]

        if local_rank <= 0:
            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f1"] = v_f1
                self.best_iou = v_miou
                self.engine.save_checkpoint(
                    os.path.join(wandb.run.dir, "best_model.pth")
                )

            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f1": v_f1}, epoch
            )

        return
