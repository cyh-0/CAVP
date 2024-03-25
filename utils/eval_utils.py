import numpy
import torch
import torchmetrics
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

def get_ml_auc(y_true, y_pred, metrics):
    score = [
        metrics(y_true[:, i], y_pred[:, i])
        for i in range(1,y_true.shape[1])
    ]
    score = numpy.asarray(score).mean()
    score = numpy.round(score, 4)
    return score


def pr_auc_score(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_performance(miou_measure_in, fg_measure_in, class_list=None):
    final_pix_metrics = miou_measure_in.get_metric_results(class_list)
    final_detect_metrics = fg_measure_in.get_metric_results(class_list)
    v_miou = final_pix_metrics[0]
    v_acc = final_pix_metrics[1]
    v_fd = final_detect_metrics[0]
    v_f1 = final_detect_metrics[1]
    v_f03 = final_detect_metrics[2]
    return v_miou, v_acc, v_fd, v_f1, v_f03


class MIoU(object):
    def __init__(self, num_classes, ignore_index, local_rank):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.local_rank = local_rank
        self.inter, self.union = 0, 0
        self.correct, self.label = 0, 0
        self.iou = numpy.array([0 for _ in range(num_classes)])
        self.acc = 0.0

    def get_metric_results(self, class_list=None):
        if class_list is None:
            return numpy.round(self.iou.mean().item(), 4), \
                   numpy.round(self.acc, 4)
        else:
            return numpy.round(self.iou[class_list].mean().item(), 4), \
                   numpy.round(self.acc, 4)

    def __call__(self, x, y):
        curr_correct, curr_label, curr_inter, curr_union = self.calculate_current_sample(x, y)
        # calculates the overall miou and acc
        self.correct = self.correct + curr_correct
        self.label = self.label + curr_label
        self.inter = self.inter + curr_inter
        self.union = self.union + curr_union

        self.acc = 1.0 * self.correct / (numpy.spacing(1) + self.label)
        self.iou = 1.0 * self.inter / (numpy.spacing(1) + self.union)
        return self.get_metric_results()

    def calculate_current_sample(self, output, target):
        # output => BxCxHxW (logits)
        # target => Bx1xHxW
        target[target == self.ignore_index] = -1
        correct, labeled = self.batch_pix_accuracy(output.data, target)
        inter, union = self.batch_intersection_union(output.data, target, self.num_classes)
        return [numpy.round(correct, 5), numpy.round(labeled, 5), numpy.round(inter, 5), numpy.round(union, 5)]

    @ staticmethod
    def batch_pix_accuracy(output, target):
        _, predict = torch.max(output, 1)

        predict = predict.int() + 1
        target = target.int() + 1

        pixel_labeled = (target > 0).sum()
        pixel_correct = ((predict == target) * (target > 0)).sum()
        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

    @ staticmethod
    def batch_intersection_union(output, target, num_class):
        _, predict = torch.max(output, 1)
        predict = predict + 1
        target = target + 1

        predict = predict * (target > 0).long()
        intersection = predict * (predict == target).long()

        area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
        area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
        area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
        return area_inter.cpu().numpy(), area_union.cpu().numpy()


class ForegroundDetect(object):
    def __init__(self, num_classes, ignore_class=255, local_rank=0):
        self.num_classes = num_classes
        self.ignore = ignore_class
        self.local_rank = local_rank
        self.confusion_matrix_ = numpy.zeros((num_classes, num_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if self.ignore is not None:
            mask = mask & (label_true != self.ignore)
        hist = numpy.bincount(
            n_class * label_true[mask].astype(int)+label_pred[mask], 
            minlength=n_class**2
        )
        hist = hist.reshape(n_class, n_class)
        return hist

    def f_beta_score(self, tp, fp, fn, beta2=1.0):
        # beta2 = beta**2
        score = ((1+beta2)*tp)/((1+beta2)*tp+beta2*fn+fp)
        return torch.nanmean(score)

    def get_metric_results(self, class_list=None):
        # the diagonal of the matrix denotes the matching true positive
        self.confusion_matrix_ = torch.tensor(self.confusion_matrix_).cuda(self.local_rank)
        tp = torch.diag(self.confusion_matrix_)
        # the vertical line is the sample shouldn't be detected but yielding positive.
        fp = self.confusion_matrix_.sum(dim=0) - tp
        # the horizontal line is the sample should be detected but yielding negative.
        fn = self.confusion_matrix_.sum(dim=1) - tp
        # the rest are tn.
        # tn = confusion_matrix_.sum() - (current_fp + current_fn + current_tp)

        if class_list is not None:
            tp = tp[class_list]
            fp = fp[class_list]
            fn = fn[class_list]

        # https://en.wikipedia.org/wiki/False_discovery_rate
        fdr = torch.nanmean(fp/(fp+tp))
        # https://en.wikipedia.org/wiki/F-score
        f1 = self.f_beta_score(tp, fp, fn, beta2=1.0)
        f_03 = self.f_beta_score(tp, fp, fn, beta2=0.3)

        fdr_score = torch.round(fdr, decimals=4).cpu().numpy()
        f1_score = torch.round(f1, decimals=4).cpu().numpy()
        f03_score = torch.round(f_03, decimals=4).cpu().numpy() 
        return fdr_score, f1_score, f03_score

    def __call__(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1).squeeze().cpu().detach().numpy()
        y = y.squeeze().cpu().detach().numpy()
        for lt, lp in zip(y, y_hat):
            self.confusion_matrix_ += self._fast_hist(lt.flatten(), lp.flatten(), self.num_classes)

