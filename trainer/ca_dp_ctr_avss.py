import os
import random
import time
from abc import abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy
import PIL
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchaudio
import torchvision
import wandb
from einops import rearrange
from loguru import logger
from PIL import Image
from tqdm import tqdm

from engine.utils import DeNormalize
from loss.contrastive_aud import ContrastLoss
from loss.losser import Losser
from utils import ddp_utils, sourcesep
from utils.eval_utils import ForegroundDetect, MIoU
from visualisation.tsne import tsne_plotter


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
        # for _, opt_group in enumerate(optimizer_a_.param_groups):
        #     opt_group["lr"] = current_lr

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

        # """ Incase we drop idx under single source setup """
        # if self.args.setup != "coco_ms":
        #     assert match_idx.sum().item() == 0

        """ change_idx: target class label"""
        mod_idx_map = {}

        for idx in change_idx:
            idx = idx.item()
            if_match[idx] = True
            true_label = img_label[idx]
            shuffle_img_label[idx] = true_label
            # curr_label = true_label[true_label != 0].detach().cpu().item()
            tmp = true_label.nonzero().squeeze(-1)
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

            # """"AVS PROCESS"""
            # frame_available = frame_available.cuda()
            # mask_available  = mask_available.cuda()
            # image = image.cuda()
            # pix_label = pix_label.cuda()
            # # image = image.view(B*frame, C, H, W)
            # mask_num = 10
            # # label = label.view(B*mask_num, H, W)
            # #! notice
            # # vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            # # gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]
            # img_label = img_label.cuda(self.local_rank, non_blocking=True)
            # # image = image[:, 0, :, :]
            # # pix_label = label[:, 0].squeeze()
            # # img_label = img_label[:,0,:]

            """END AVS PROCESS"""
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
            output_cat, ctr_feature_cat, attn_v = model_v_(image, audio, shuffle_info, ow_flag)

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

            l_ce, l_ctr_global, l_ctr_pixel, _, l_mctr_av = self.loss_func(
                output, pix_label
            )

            loss = l_ce  + l_ctr_av #+ l_ctr_global + l_ctr_pixel + l_mctr_av

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
            del img_label, waveform #, audio

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
                            "lr/lr_bkb": lr_bkb,
                            "lr/lr_attn": lr_attn,
                            "lr/lr_seg": lr_seg,
                            "lr/lr_audio": lr_audio,
                        }
                    )
                if batch_idx % self.args.upload_iter == 0:
                    # attn_v = attn_v[:B]
                    # attn_map = rearrange(attn_v.clone().squeeze().unsqueeze(-1).sum(-1).sum(-1).unsqueeze(-1), 'b n (h w) c -> b n c h w', h=128, w=128, b=B)
                    self.vis_tool.upload_wandb_image(
                        image.cpu().detach(),
                        pix_label.cpu().detach(),
                        output.cpu().detach(),
                        torch.softmax(output, dim=1).cpu().detach(),
                        # heatmap=attn_map.cpu().detach(),
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
        target_list, audio_fea_list, fusion_fea_list = [], [], []
        miou_measure = MIoU(
            num_classes=self.args.num_classes,
            ignore_index=255,
            local_rank=self.local_rank,
        )
        fg_measure = ForegroundDetect(num_classes=71, local_rank=self.local_rank)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        # from math import ceil

        for batch_idx, batch_data in enumerate(tbar):
            # continue
            # image, waveform, label, img_label, vid_temporal_mask_flag, gt_temporal_mask_flag, mask_num, video_name = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            image, waveform, pix_label, img_label, name, frame_available, mask_available = batch_data
            mask_num = mask_available.sum().long().item()
            B, T, C, H, W = image.shape
            image = image.cuda(non_blocking=True)
            waveform = waveform.cuda(non_blocking=True).view(B, T, 1, -1)
            img_label = img_label.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)
            
            """BATCH OPERATION"""
            # curr_image = image[0, :mask_num]
            # curr_img_label = img_label[0, :mask_num]
            # curr_pix_label = pix_label[0, :mask_num]
            # curr_waveform = waveform[0, :mask_num]
            # curr_audio = self.preprocess_audio(curr_waveform)
            # avl_map_logits,_ = model_v_.module(curr_image, curr_audio, eval_mode=True)
            # for i in range(avl_map_logits.shape[0]):
            #     start = time.time()
            #     # fg_measure(avl_map_logits/[i, None], curr_pix_label[i])
            #     miou, acc = miou_measure(avl_map_logits[i, None], curr_pix_label[i])
            #     tbar.set_description("epoch ({}/) | mIoU {} Acc {}".format(epoch, miou, acc))
            #     print("Time: ", time.time() - start)
                
            for i in range(mask_num):
                curr_image = image[0, i, None]
                curr_img_label = img_label[0, i]
                curr_pix_label = pix_label[0, i]
                curr_waveform = waveform[0, i, None]
                """
                Get Pred
                """
                # audio = self.preprocess_audio(waveform)
                curr_audio = self.preprocess_audio(curr_waveform)
                if self.args.use_baseline:
                    avl_map_logits = model_v_.module(curr_image)
                else:
                    (
                        avl_map_logits,
                        fusion_fea,
                    ) = model_v_.module(curr_image, curr_audio, eval_mode=True)
                """
                Prepare T-SNE
                """
                fea = model_v_.module.audio_backbone(curr_audio)
                # fea = model_v_.module.audio_proj(fea)
                tar = img_label[0, i]
                tar[0] = 0
                if tar.sum() == 1:
                    class_id = tar.cpu().nonzero().squeeze()
                    gt_ = torch.nn.functional.interpolate(curr_pix_label[None,None,:].float(), size=fusion_fea.shape[2:],
                                                mode='nearest').long().squeeze()
                    class_fea = fusion_fea[:,:, gt_==class_id].squeeze().permute(1,0).cpu()
                    if class_fea.shape[0] > 5:
                        sample_num = 5
                        target_list.append(class_id)
                        audio_fea_list.append(fea.cpu())
                        fusion_fea_list.append(class_fea[torch.randint(0, class_fea.shape[0], size=(sample_num,)),:])


                """
                Prepare Metrics
                """
                fg_measure(avl_map_logits, curr_pix_label)
                miou, acc = miou_measure(avl_map_logits, curr_pix_label)
                tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))


            #     self.vis_tool.upload_wandb_image(
            #             curr_image.cpu().detach(),
            #             curr_pix_label.cpu().detach(name).unsqueeze(0),
            #             avl_map_logits.cpu().detach(),
            #             torch.softmax(avl_map_logits, dim=1).cpu().detach(),
            #             status="test",
            #         )
            # a=1

        final_pix_metrics = miou_measure.get_metric_results()
        final_detect_metrics = fg_measure.get_metric_results()

        v_miou = final_pix_metrics[0]
        v_acc = final_pix_metrics[1]

        v_fd = final_detect_metrics[0]
        v_f1 = final_detect_metrics[1]
        v_f03 = final_detect_metrics[2]

        if self.local_rank <= 0:
            logger.success(
                f"| mIoU: {v_miou:.4f} | acc: {v_acc:.4f} | fdr: {v_fd:.4f} | f_1: {v_f1:.4f} | f_0.3: {v_f03:.4f}"
            )
            audio_fea_list = torch.cat(audio_fea_list, dim=0)
            fusion_fea_list = torch.stack(fusion_fea_list, dim=0)
            target_list = torch.stack(target_list)
            tsne_plotter(audio_fea_list, target_list, target_list, num_classes=(self.args.num_classes-1), name="audio_fea")
            target_list = torch.repeat_interleave(target_list[:,None], 5, dim=1)
            tsne_plotter(
                rearrange(fusion_fea_list, "b n d -> (b n) d"), 
                target_list.view(-1), 
                target_list, 
                num_classes=(self.args.num_classes-1),
                name="fusion_fea"
            )

            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f_1"] = v_f1
                wandb.run.summary["best_f_0.3"] = v_f03
                self.best_iou = v_miou
                self.engine.save_checkpoint(
                    os.path.join(wandb.run.dir, "best_model.pth")
                )

            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f_1": v_f1,
                 "f_0.3": v_f03},
                epoch
            )

        return

    @torch.no_grad()
    def test(self, model_v_, model_a_, epoch, test_loader, status="test"):
        model_v_.eval()
        model_a_.eval()
        target_list, audio_fea_list, c_list = [], [], []
        miou_measure = MIoU(
            num_classes=self.args.num_classes,
            ignore_index=255,
            local_rank=self.local_rank,
        )
        fg_measure = ForegroundDetect(num_classes=71, local_rank=self.local_rank)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        # from math import ceil

        for batch_idx, batch_data in enumerate(tbar):
            image, waveform, label, img_label, vid_temporal_mask_flag, gt_temporal_mask_flag, mask_num, video_name = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            B, frame, C, H, W = image.shape
            # waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            image = image[0]
            label = label[0]
            img_label = img_label[0]
            for i in range(mask_num.item()):
                """"AVS PROCESS"""
                # if gt_temporal_mask_flag.squeeze()[i] == 0:
                    # continue

                
                curr_img_label = img_label.cuda(self.local_rank, non_blocking=True)
                curr_image = image[i, :, :].unsqueeze(0)
                curr_pix_label = label[i].squeeze()
                curr_img_label = img_label[i,:]
                curr_a_fea = model_v_.module.audio_backbone(waveform).view(B,frame,-1)[:,i,:]
                """END AVS PROCESS"""

                """
                Get Pred
                """
                # audio = self.preprocess_audio(waveform)
                if self.args.use_baseline:
                    avl_map_logits = model_v_.module(image)
                else:
                    (
                        avl_map_logits,
                        _,
                    ) = model_v_.module(curr_image, curr_a_fea, eval_mode=True)
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
                fg_measure(avl_map_logits, curr_pix_label)
                miou, acc = miou_measure(avl_map_logits, curr_pix_label)
                tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))
                
                self.vis_tool.upload_wandb_image(
                        curr_image.cpu().detach(),
                        curr_pix_label.cpu().detach().unsqueeze(0),
                        avl_map_logits.cpu().detach(),
                        torch.softmax(avl_map_logits, dim=1).cpu().detach(),
                        status="test",
                        # folder=video_name,
                    )

        final_pix_metrics = miou_measure.get_metric_results()
        final_detect_metrics = fg_measure.get_metric_results()

        v_miou = final_pix_metrics[0]
        v_acc = final_pix_metrics[1]
        
        v_fd = final_detect_metrics[0]
        v_f1 = final_detect_metrics[1]
        v_f03 = final_detect_metrics[2]

        if self.local_rank <= 0:
            logger.success(
                f"| mIoU: {v_miou:.4f} | acc: {v_acc:.4f} | fdr: {v_fd:.4f} | f_1: {v_f1:.4f} | f_0.3: {v_f03:.4f}"
            )
            # fea_list = torch.cat(audio_fea_list, dim=0)
            # target_list = torch.stack(target_list)
            # tsne_plotter(fea_list, target_list, c_list, num_classes=(self.args.num_classes-1), name="audio_fea")
            if v_miou > self.best_iou:
                wandb.run.summary["best_epoch"] = epoch
                wandb.run.summary["best_miou"] = v_miou
                wandb.run.summary["best_acc"] = v_acc
                wandb.run.summary["best_fd"] = v_fd
                wandb.run.summary["best_f_1"] = v_f1
                wandb.run.summary["best_f_0.3"] = v_f03
                self.best_iou = v_miou
                self.engine.save_checkpoint(
                    os.path.join(wandb.run.dir, "best_model.pth")
                )

            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f_1": v_f1, "f_0.3": v_f03},
                epoch
            )

        return