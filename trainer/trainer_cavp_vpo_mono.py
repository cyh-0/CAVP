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
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from models.cavp_model import SoundBank


class CAVP_TRAINER:
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
        self.sound_bank = SoundBank(
            out_dim=16000*int(args.audio_len), args=args, device=args.local_rank
        )

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
        lr_bkb = optimizer_v_.param_groups[0]["lr"]
        lr_attn = optimizer_v_.param_groups[2]["lr"]
        lr_seg = optimizer_v_.param_groups[4]["lr"]
        lr_audio = self.args.lr
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

        """ change_idx: target class label"""
        mod_idx_map = {}

        for idx in change_idx:
            idx = idx.item()
            if_match[idx] = True
            true_label = img_label[idx]
            shuffle_img_label[idx] = true_label
            tmp = true_label.nonzero().squeeze(-1)
            curr_label = tmp[tmp.nonzero().squeeze()]
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
            shuffle_idx = torch.randperm(image.shape[0]).cuda(self.local_rank)
            shuffle_img_label = img_label.clone()[shuffle_idx]
            shuffle_pix_label = pix_label.clone()[shuffle_idx]
            if_match = torch.all(torch.eq(img_label, shuffle_img_label), dim=1)
            """"1"""
            shuffle_audio = waveform.clone()[shuffle_idx]
            mod_idx_map = None
            if epoch >= 1:
                if_match, shuffle_img_label, mod_idx_map = self.overwrite_miss_match(
                    if_match, shuffle_img_label, img_label
                )
                if ow_flag:
                    shuffle_audio = self.sound_bank.overwrite_audio_feature(
                        shuffle_audio, waveform, mod_idx_map
                    )

            """"1"""
            self.sound_bank.update_bank(waveform, img_label)
            input_waveform = torch.cat((waveform, shuffle_audio))
            audio = self.preprocess_audio(input_waveform)
            output_cat, ctr_feature_cat, pack_ = model_v_(image, audio, None, ow_flag)

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

            l_ce = self.loss_func(output, pix_label)

            loss = l_ce + l_ctr_av

            loss.backward()
            optimizer_v_.step()
            optimizer_a_.step()

            del shuffle_idx, shuffle_pix_label, shuffle_img_label,
            del ctr_feature_shuff, ctr_feature

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
                    "epoch {}:  l_ce {:.3f}  l_ctr_av {:.3f}".format(
                        epoch,
                        l_ce.item(),
                        l_ctr_av.item(),
                    )
                )
                if batch_idx % self.args.display_iter == 0:
                    self.vis_tool.upload_metrics(
                        {
                            "loss/loss": loss,
                            "loss/cross_entropy": l_ce.item(),
                            "loss/l_ctr_av": l_ctr_av.item(),
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
            del image, output, img_label, pix_label, waveform, shuffle_audio
            del output_cat, ctr_feature_cat, pack_
            del l_ce, l_ctr_av
            
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
        for batch_idx, (image, waveform, pix_label, img_label, name) in enumerate(tbar):
            waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)
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
                    _,
                ) = model_v_.module(image, audio, eval_mode=True)
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
        v_f03 = final_detect_metrics[2]

        if self.local_rank <= 0:
            logger.success(
                f"| mIoU: {v_miou:.4f} | acc: {v_acc:.4f} | fdr: {v_fd:.4f} | f1: {v_f1:.4f} | f_0.3: {v_f03:.4f}"
            )
            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f1"] = v_f1
                wandb.run.summary["best_f_0.3"] = v_f03
                self.best_iou = v_miou
                self.engine.save_checkpoint(
                    os.path.join(wandb.run.dir, "best_model.pth")
                )
            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f_0.3": v_f03, "f1": v_f1},
                epoch
            )
        return
