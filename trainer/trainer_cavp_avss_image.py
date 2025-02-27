import os
import random
import time
from abc import abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy
import PIL
import torch, torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchaudio
from einops import rearrange
from loguru import logger
from PIL import Image
from tqdm import tqdm

import wandb
from engine.utils import DeNormalize
from loss.contrastive_aud import ContrastLoss
from loss.losser import Losser
from utils import ddp_utils, sourcesep
from utils.eval_utils import ForegroundDetect, MIoU
from utils.eval_utils import get_performance
import pandas as pd
from models.cavp_model import SoundBank

class CAVP_TRAINER:
    def __init__(self, args, train_loader, visual_tool, engine, lr_scheduler) -> None:
        self.args = args
        self.engine = engine
        self.vis_tool = visual_tool
        self.local_rank = self.engine.local_rank
        self.bcel_loss = nn.BCEWithLogitsLoss()
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
        self.loss_mse = nn.MSELoss()
        self.ow_rate = args.ow_rate
        self.eval_list = pd.read_csv("./utils/eval_list.txt", skip_blank_lines=False)
        self.eval_list = self.eval_list.iloc[:, 0].tolist()
        self.win_size = 2
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
        # Segmentation
        for _, opt_group in enumerate(optimizer_v_.param_groups[:8]):
            opt_group["lr"] = current_lr * 10.0
        # Backbone + Cross Attn
        for _, opt_group in enumerate(optimizer_v_.param_groups[8:]):
            opt_group["lr"] = current_lr
        # # Audio
        lr_seg = optimizer_v_.param_groups[0]["lr"]
        lr_bkb = optimizer_v_.param_groups[9]["lr"]
        lr_attn = optimizer_v_.param_groups[11]["lr"]
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

    def visualise_synthetic(self, model_v_, curr_image, curr_waveform, curr_pix_label, name="", status=""):
        curr_audio = self.preprocess_audio(curr_waveform)
        synthetic_output, _, _ = \
        model_v_.module(curr_image, curr_audio, eval_mode=True)
        self.vis_tool.upload_wandb_image(
            curr_image.cpu().detach(),
            curr_pix_label.cpu().detach().unsqueeze(0),
            synthetic_output.cpu().detach(),
            torch.softmax(synthetic_output, dim=1).cpu().detach(),
            status=status,
            folder=name,
            show_y=False,
        )
        return

    @abstractmethod
    def train(
        self, model_v_, model_a_, optimizer_v_, optimizer_a_, epoch, train_loader
    ):
        model_v_.train()
        model_a_.train()
        loader_len = len(train_loader)

        if self.local_rank <= 0:
            tbar = tqdm(train_loader, total=loader_len, ncols=120, position=0, leave=True)
        else:
            tbar = train_loader

        ow_flag = True if epoch >= 1 else False

        # for batch_idx, (image, waveform, pix_label, img_label, _) in enumerate(tbar):
        for batch_idx, batch_data in enumerate(tbar):
            self.engine.update_iteration(epoch, epoch * loader_len + batch_idx)

            image, waveform, pix_label, img_label, name, frame_available, mask_available = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            B, T, C, H, W = image.shape
            waveform = waveform.view(B, T, -1)
            available =  ((frame_available + mask_available) == 2)
            avail_idx = [available[i].nonzero() for i in range(B)]
            sel_idx = [random.choice(avail_idx[i]).item() for i in range(B)]

            image = image[range(B), sel_idx, :, :]
            pix_label = pix_label[range(B), sel_idx,:, :]
            img_label = img_label[range(B), sel_idx,:]
            waveform = waveform[range(B), sel_idx, None]

            """END AVS PROCESS"""
            waveform = waveform.cuda(self.local_rank, non_blocking=True)
            image = image.cuda(self.local_rank, non_blocking=True)
            pix_label = pix_label.cuda(self.local_rank, non_blocking=True)
            img_label = img_label.cuda(self.local_rank, non_blocking=True)

            optimizer_v_.zero_grad()
            optimizer_a_.zero_grad()

            shuffle_idx = torch.randperm(image.shape[0]).cuda(self.local_rank)
            shuffle_img_label = img_label.clone()[shuffle_idx]
            shuffle_pix_label = pix_label.clone()[shuffle_idx]
            if_match = torch.all(torch.eq(img_label, shuffle_img_label), dim=1)
            """"1"""
            shuffle_audio = waveform.clone()[shuffle_idx]
            mod_idx_map = None
            
            if self.args.avsbench_split == "all" and self.args.setup != "avss_binary":
                if epoch >= 1:
                    if_match, shuffle_img_label, mod_idx_map = self.overwrite_miss_match(
                        if_match, shuffle_img_label, img_label
                    )
                    """ Update memory bank """
                    if ow_flag:
                        shuffle_audio = self.sound_bank.overwrite_audio_feature(
                            shuffle_audio, waveform, mod_idx_map
                        )
                self.sound_bank.update_bank(waveform, img_label)

            input_waveform = torch.cat((waveform, shuffle_audio))
            
            audio = self.preprocess_audio(input_waveform)
            output_cat, ctr_feature_cat, pack_ = model_v_(image, audio, None, ow_flag)

            B = image.shape[0]
            output = output_cat[:B] + output_cat[B:] * 0.0
            ctr_feature = ctr_feature_cat[:B]
            ctr_feature_shuff = ctr_feature_cat[B:]
            assert ctr_feature.shape[0] == ctr_feature_shuff.shape[0]

            # if the shuffle audio don't match the original sources, we give background.
            shuffle_pix_label[~if_match] = 0
            # if the shuffle audio match the original sources, we give its gt.
            shuffle_pix_label[if_match] = pix_label[if_match]

            l_ctr_av = self.loss_ctr_av(
                ctr_feature, pix_label, ctr_feature_shuff, shuffle_pix_label
            )

            l_ce = self.loss_func(
                output, pix_label, pack_
            )

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
                if batch_idx % self.args.upload_iter * 500 == 0:
                    attn_v = None
                    self.vis_tool.upload_wandb_image(
                        image.cpu().detach(),
                        pix_label.cpu().detach(),
                        output.cpu().detach(),
                        torch.softmax(output, dim=1).cpu().detach(),
                        heatmap=attn_v,
                        status="train",
                        caption=name,
                    )
                    
            del image, output, img_label, pix_label, waveform, shuffle_audio
            del output_cat, ctr_feature_cat, pack_
            del loss, l_ce, l_ctr_av

        return

    @torch.no_grad()
    def validation(self, model_v_, model_a_, epoch, test_loader, status="val"):
        model_v_.eval()
        model_a_.eval()

        target_list, visual_fea_list, audio_fea_list, fusion_fea_list = [], [], [], []
        bg_fea_list = []

        miou_measure = MIoU(num_classes=self.args.num_classes, ignore_index=255, local_rank=self.local_rank)
        miou_measure_ms = MIoU(num_classes=self.args.num_classes, ignore_index=255, local_rank=self.local_rank)
        fg_measure = ForegroundDetect(num_classes=self.args.num_classes, local_rank=self.local_rank)
        fg_measure_ms = ForegroundDetect(num_classes=self.args.num_classes, local_rank=self.local_rank)

        tbar = tqdm(test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True)

        for batch_idx, batch_data in enumerate(tbar):
            # continue
            # image, waveform, label, img_label, vid_temporal_mask_flag, gt_temporal_mask_flag, mask_num, video_name = \
            # batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            image, waveform, pix_label, img_label, name, frame_available, mask_available = batch_data
            mask_num = mask_available.sum().long().item()
            B, T, C, H, W = image.shape
            image = image.cuda(non_blocking=True)
            waveform = waveform.cuda(non_blocking=True).view(B, T, 1, -1)
            img_label = img_label.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)

            for i in range(mask_num):
                curr_image = image[0, i, None]
                curr_img_label = img_label[0, i]
                curr_pix_label = pix_label[0, i]
                curr_waveform = waveform[0, i, None]
                curr_audio = self.preprocess_audio(curr_waveform)
                avl_map_logits, fusion_fea, fea = \
                    model_v_.module(curr_image, curr_audio, eval_mode=True)                
                """
                Prepare Metrics
                """
                fg_measure(avl_map_logits, curr_pix_label)
                miou, acc = miou_measure(avl_map_logits, curr_pix_label)

                uniq_id, count = curr_pix_label.unique(return_counts=True)
                valid_id = uniq_id[count > 100]
                if valid_id.shape[0] > 2:
                    fg_measure_ms(avl_map_logits, curr_pix_label)
                    miou_ms, acc_ms = miou_measure_ms(avl_map_logits, curr_pix_label)
                    if name[0] in self.eval_list:
                        self.vis_tool.upload_wandb_image(
                            curr_image.cpu().detach(),
                            curr_pix_label.cpu().detach().unsqueeze(0),
                            avl_map_logits.cpu().detach(),
                            torch.softmax(avl_map_logits, dim=1).cpu().detach(),
                            status="test",
                            folder=name[0],
                            caption=name[0],
                        )
                else:
                    miou_ms, acc_ms = miou_measure_ms.get_metric_results()

                tbar.set_description(
                    "epoch ({}) |ALL| mIoU {} Acc {} |MS| mIoU {} Acc {}".format(
                        epoch, miou, acc, miou_ms, acc_ms
                    )
                )

        class_list = None
        v_miou, v_acc, v_fd, v_f1, v_f03 = get_performance(miou_measure, fg_measure, class_list=class_list)
        v_miou_ms, v_acc_ms, v_fd_ms, v_f1_ms, v_f03_ms = get_performance(
            miou_measure_ms, fg_measure_ms,class_list=class_list
        )

        if self.local_rank <= 0:
            logger.success(
                f"|ALL| mIoU: {v_miou:.4f} | acc: {v_acc:.4f} | fdr: {v_fd:.4f} | f_1: {v_f1:.4f} | f_0.3: {v_f03:.4f}"
            )
            logger.success(
                f"|MS| mIoU: {v_miou_ms:.4f} | acc: {v_acc_ms:.4f} | fdr: {v_fd_ms:.4f} | f_1: {v_f1_ms:.4f} | f_0.3: {v_f03_ms:.4f}"
            )
            """ 
                Plot T-SNE 
            """
            list_of_feature = (audio_fea_list, visual_fea_list, fusion_fea_list, bg_fea_list, target_list)
            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                # ALL
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f_1"] = v_f1
                wandb.run.summary["best_f_0.3"] = v_f03
                # MS
                wandb.run.summary["best_miou_ms"] = v_miou_ms
                wandb.run.summary["best_acc_ms"] = v_acc_ms
                wandb.run.summary["best_fd_ms"] = v_fd_ms
                wandb.run.summary["best_f_1_ms"] = v_f1_ms
                wandb.run.summary["best_f_0.3_ms"] = v_f03_ms
                #
                self.best_iou = v_miou
                if not self.args.ignore_ckpt:
                    self.engine.save_checkpoint(
                        os.path.join(wandb.run.dir, "best_model.pth")
                    )

            self.vis_tool.upload_metrics(
                {
                    # ALL
                    "miou": v_miou,
                    "acc": v_acc,
                    "fdr": v_fd,
                    "f_1": v_f1,
                    "f_0.3": v_f03,
                    # MS
                    "multi-source/miou_ms": v_miou_ms,
                    "multi-source/acc_ms": v_acc_ms,
                    "multi-source/fdr_ms": v_fd_ms,
                    "multi-source/f_1_ms": v_f1_ms,
                    "multi-source/f_0.3_ms": v_f03_ms,
                },
                epoch,
            )
        del target_list, visual_fea_list, audio_fea_list, fusion_fea_list, bg_fea_list
        del image, waveform, pix_label, img_label, name, frame_available, mask_available
        del curr_image, curr_audio, curr_img_label, curr_pix_label
        del avl_map_logits, fusion_fea, fea
        del v_miou, v_acc, v_fd, v_f1, v_f03
        del v_miou_ms, v_acc_ms, v_fd_ms, v_f1_ms, v_f03_ms
        return

        