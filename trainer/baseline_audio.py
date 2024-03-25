import os
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchaudio
import wandb
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from engine.utils import DeNormalize
from loss.contrastive_aud import ContrastLoss
from loss.losser import Losser
from utils import sourcesep
from utils.eval_utils import ForegroundDetect, MIoU
from visualisation.tsne import tsne_plotter

# rewrite replace_value to for loop



def replace_value(tensor):
    # replace tensor value at 2 and 3 to 0 and 1 at once
    a = torch.zeros_like(tensor)
    a[tensor==0] = 0
    b = torch.zeros_like(tensor)
    b[tensor==1] = 1
    c = torch.zeros_like(tensor)
    c[tensor==2] = 3
    d = torch.zeros_like(tensor)
    d[tensor==3] = 4
    e = torch.zeros_like(tensor)
    e[tensor==4] = 5
    f = torch.zeros_like(tensor)
    f[tensor==5] = 6
    g = torch.zeros_like(tensor)
    g[tensor==6] = 7
    h = torch.zeros_like(tensor)
    h[tensor==7] = 8
    i = torch.zeros_like(tensor)
    i[tensor==8] = 9
    j = torch.zeros_like(tensor)
    j[tensor==9] = 10
    k = torch.zeros_like(tensor)
    k[tensor==10] = 12
    l = torch.zeros_like(tensor)
    l[tensor==11] = 13
    m = torch.zeros_like(tensor)
    m[tensor==13] = 15
    n = torch.zeros_like(tensor)
    n[tensor==14] = 16
    o = torch.zeros_like(tensor)
    o[tensor==15] = 17
    p = torch.zeros_like(tensor)
    p[tensor==16] = 18
    q = torch.zeros_like(tensor)
    q[tensor==17] = 19
    r = torch.zeros_like(tensor)
    r[tensor==18] = 20
    s = torch.zeros_like(tensor)
    s[tensor==19] = 21
    t = torch.zeros_like(tensor)
    t[tensor==12] = 0
    out = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + r + s
    return out + t


class BASELINE:
    def __init__(self, args, train_loader, visual_tool, engine, lr_scheduler) -> None:
        self.args = args
        self.engine = engine
        self.vis_tool = visual_tool
        self.local_rank = self.engine.local_rank
        # self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
        self.criterion = nn.BCEWithLogitsLoss()
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

    def lr_step(self, optimizer_v_, current_lr, use_baseline=False):
        lr_bkb, lr_attn, lr_ctr, lr_seg = 0.0, 0.0, 0.0, 0.0
        if use_baseline:
            optimizer_v_.param_groups[0]["lr"] = current_lr
            optimizer_v_.param_groups[1]["lr"] = current_lr
            for _, opt_group in enumerate(optimizer_v_.param_groups[2:]):
                opt_group["lr"] = current_lr * 10.0
            lr_bkb = optimizer_v_.param_groups[0]["lr"]
            lr_seg = optimizer_v_.param_groups[2]["lr"]
        else:
            # Backbone + Cross Attn
            for _, opt_group in enumerate(optimizer_v_.param_groups[:4]):
                opt_group["lr"] = current_lr
            # Contrast Proj
            # for _, opt_group in enumerate(optimizer_v_.param_groups[4:8]):
            #     opt_group["lr"] = current_lr * 1
            # Segmentation
            for _, opt_group in enumerate(optimizer_v_.param_groups[4:]):
                opt_group["lr"] = current_lr * 10.0
            lr_bkb = optimizer_v_.param_groups[0]["lr"]
            lr_attn = optimizer_v_.param_groups[2]["lr"]
            lr_seg = optimizer_v_.param_groups[8]["lr"]
        return lr_bkb, lr_attn, lr_seg

    def overwrite_miss_match(self, if_match, shuffle_img_label, img_label):
        false_list = (if_match == 0).nonzero(as_tuple=True)[0]
        n_false = false_list.shape[0]
        shuffle_idx = torch.randperm(n_false)[: (n_false // 2)]
        change_idx = false_list[shuffle_idx]

        # tmp_audio_idx = [0] * self.num_classes

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
        model_v_.eval()
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
            output = model_a_.module.forward_cls(audio)
            l_ce = self.criterion(output,img_label.float())
            l_ce.backward()
            optimizer_a_.step()


            lr_bkb, lr_attn, lr_seg = 0.0, 0.0, 0.0

            if self.local_rank <= 0:
                tbar.set_description(
                    "epoch {}:  l_ce {:.3f} ".format(
                        epoch,
                        l_ce.item(),
                    )
                )
                if batch_idx % self.args.display_iter == 0:
                    self.vis_tool.upload_metrics(
                        {
                            # "loss/loss": l_ce,
                            "loss/cross_entropy": l_ce.item(),
                            "lr/lr_audio": optimizer_a_.param_groups[0]["lr"],
                            "lr/lr_bkb": lr_bkb,
                            "lr/lr_attn": lr_attn,
                            "lr/lr_seg": lr_seg,
                        }
                    )

            del image, output, pix_label, l_ce
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
        # fg_measure = ForegroundDetect(num_classes=24, local_rank=self.local_rank)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        # from math import ceil
        pred_list, target_list = [],[]
        for batch_idx, (image, waveform, pix_label, img_label, name) in enumerate(tbar):

            waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)
            # index_table = test_loader.dataset.dataset_v.index_table
            """
            Get Pred
            """
            audio = self.preprocess_audio(waveform)
            avl_map_logits = model_v_.module(image)
            # """remove column index 12"""
            # avl_map_logits = torch.cat(
            #     [avl_map_logits[:, :12], avl_map_logits[:, 13:]], dim=1
            # )
            
            output_a = model_a_.module.forward_cls(audio)
            # output_a_logits = torch.cat([output_a[:, :2], output_a[:, 3:11], output_a[:, 12:14], output_a[:, 15:]], dim=1)
            # new_image_label = torch.cat([img_label[:, :2], img_label[:, 3:11], img_label[:, 12:14], img_label[:, 15:]], dim=1)
            output_a_prob = torch.sigmoid(output_a)
            pred_list.append(output_a_prob[0])
            target_list.append(img_label[0])
            output_a_pred = (output_a_prob > 0.5).float() * 1

            no_sound_idx = (output_a_pred == 0).squeeze().nonzero().tolist()
            pred = avl_map_logits.max(dim=1)[1]
            seg_pred = replace_value(pred)
            for idx in no_sound_idx:
                seg_pred[seg_pred == idx[0]] = 0            # give zero 

            # print("//")
            # print(torch.unique(pred), torch.unique(pix_label))
            # print(torch.unique(seg_pred))


            # # pred[gt == 255] = 255
            # from utils.tensor_board import colorize_mask
            # pred_lists = [colorize_mask(i, self.vis_tool.pallete) for i in pred.cpu().numpy()]
            # # pred_list[0].show()

            # pred = pix_label
            # pred_lists = [colorize_mask(i, self.vis_tool.pallete) for i in pred.cpu().numpy()]
            # # pred_list[0].show()
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
            # fg_measure(avl_map_logits, pix_label)
            miou, acc = miou_measure(seg_pred, pix_label)
            # miou, acc = miou_measure(avl_map_logits, pix_label)
            tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))

        from sklearn.metrics import average_precision_score, roc_auc_score
        all_preds = torch.cat(pred_list).cpu().numpy()
        all_gts = torch.cat(target_list).cpu().numpy()
        all_auc = roc_auc_score(all_gts, all_preds)
        print("AUDIO CLS AUC: ", all_auc)

        final_pix_metrics = miou_measure.get_metric_results()
        # final_detect_metrics = fg_measure.get_metric_results()

        v_miou = final_pix_metrics[0]
        v_acc = final_pix_metrics[1]
        v_fd = 0
        v_f1 = 0

        if self.local_rank <= 0:
            logger.success(
                f"| mIoU: {v_miou:.2f} | acc: {v_acc:.2f} | fdr: {v_fd:.2f} | f1: {v_f1:.2f}"
            )
            self.vis_tool.upload_metrics(
                {"miou": v_miou, "acc": v_acc, "fdr": v_fd, "f1": v_f1}, epoch
            )

        return

    @torch.no_grad()
    def eval(self, model_v_, model_a_, epoch, test_loader, local_rank, status="val"):
        model_v_.eval()
        model_a_.eval()
        target_list, audio_fea_list, c_list = [], [], []

        miou_measure = MIoU(num_classes=self.args.num_classes, ignore_index=255)
        fg_measure = ForegroundDetect(num_classes=24)
        tbar = tqdm(
            test_loader, total=int(len(test_loader)), ncols=120, position=0, leave=True
        )
        index_table = test_loader.dataset.dataset_v.index_table

        # from math import ceil
        for batch_idx, (image, waveform, pix_label, img_label, name) in enumerate(tbar):
            waveform = waveform.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            pix_label = pix_label.cuda(non_blocking=True)

            audio = self.preprocess_audio(waveform)
            fea = model_v_.module.audio_backbone(audio)
            fea = model_v_.module.audio_proj(fea)
            tar = img_label[0]
            tar = tar[tar != 0]
            target_list.append(tar)
            audio_fea_list.append(fea)
            c_list.append(index_table[tar.item()])
            continue

            # audio_feature = model_v_.audio_backbone(audio)
            (
                avl_map_logits,
                _,
                _,
            ) = model_v_(image, audio)

            fg_measure(avl_map_logits, pix_label)
            miou, acc = miou_measure(avl_map_logits, pix_label)
            tbar.set_description("epoch ({}) | mIoU {} Acc {}".format(epoch, miou, acc))

            self.vis_tool.upload_wandb_image(
                image.cpu().detach(),
                pix_label.cpu().detach(),
                avl_map_logits.cpu().detach(),
                torch.softmax(avl_map_logits, dim=1).cpu().detach(),
                status=status,
            )

        fea_list = torch.cat(audio_fea_list, dim=0)
        target_list = torch.stack(target_list)

        tsne_plotter(fea_list, target_list, c_list, name="audio_fea")

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
