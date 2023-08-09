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


class MIoU(object):
    def __init__(self, num_classes, ignore_index, local_rank):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.local_rank = local_rank
        self.inter, self.union = 0, 0
        self.correct, self.label = 0, 0
        self.iou = numpy.array([0 for _ in range(num_classes)])
        self.acc = numpy.array([0 for _ in range(num_classes)])

    def get_metric_results(self):
        return numpy.round(self.iou.mean().item(), 4), numpy.round(self.acc, 4)

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
        # self.confusion_matrix_1 = torch.zeros((num_classes, num_classes)).cuda(local_rank)
        self.cal_cf_matrix = torchmetrics.ConfusionMatrix(num_classes=self.num_classes, task="multiclass",
                                                          ignore_index=ignore_class).cuda(local_rank)

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        mask &= (label_pred >= 0) & (label_pred < n_class)

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

    def get_metric_results(self):
        # the diagonal of the matrix denotes the matching true positive
        self.confusion_matrix_ = torch.tensor(self.confusion_matrix_).cuda(self.local_rank)
        tp = torch.diag(self.confusion_matrix_)
        # the vertical line is the sample shouldn't be detected but yielding positive.
        fp = self.confusion_matrix_.sum(dim=0) - tp
        # the horizontal line is the sample should be detected but yielding negative.
        fn = self.confusion_matrix_.sum(dim=1) - tp
        # the rest are tn.
        # tn = confusion_matrix_.sum() - (current_fp + current_fn + current_tp)

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

        # self.confusion_matrix_1 += self.cal_cf_matrix(torch.argmax(y_hat, dim=1).flatten().cuda(self.local_rank),
        #                                              y.flatten().cuda(self.local_rank))

#
# class ForegroundDetect(object):
#     def __init__(self, num_classes, ignore_classes):
#         self.in_scores = []
#         self.out_scores = []
#         self.ap = .0
#         self.roc = .0
#         self.fpr = .0
#
#     def get_metric_results(self):
#         curr_auroc, curr_aupr, curr_fpr = self.calculate_metrics(numpy.array(self.out_scores), numpy.array(self.in_scores))
#         self.ap = numpy.mean(curr_aupr)
#         self.roc = numpy.mean(curr_auroc)
#         self.fpr = numpy.mean(curr_fpr)
#         return numpy.round(self.fpr, 4), numpy.round(self.ap, 4), numpy.round(self.roc, 4)
#
#     def __call__(self, y_hat, y):
#         # calculates the confident based on the foreground softmax score
#         # scores = torch.sum(torch.softmax(y_hat, dim=1)[:, 1:], dim=1)
#         scores = y_hat * 1.
#         # flatten the score
#         self.in_scores += scores[y == 0].flatten().cpu().tolist()
#         self.out_scores += scores[y > 0].flatten().cpu().tolist()
#         return 0, 0, 0
#         # calculate current metrics
#
#     def calculate_metrics(self, _pos, _neg, recall_level=0.95):
#         pos = numpy.array(_pos[:]).reshape((-1, 1))
#         neg = numpy.array(_neg[:]).reshape((-1, 1))
#         examples = numpy.squeeze(numpy.vstack((pos, neg)))
#         labels = numpy.zeros(len(examples), dtype=numpy.int32)
#         labels[:len(pos)] += 1
#         auroc_ = sklearn.metrics.roc_auc_score(labels, examples)
#         aupr_ = sklearn.metrics.average_precision_score(labels, examples)
#         fpr_ = self.fpr_at_recall(labels, examples, recall_level)
#         return auroc_, aupr_, fpr_
#
#     def fpr_at_recall(self, y_true, y_score, recall_level, pos_label=None):
#         classes = numpy.unique(y_true)
#         if (pos_label is None and
#                 not (numpy.array_equal(classes, [0, 1]) or
#                      numpy.array_equal(classes, [-1, 1]) or
#                      numpy.array_equal(classes, [0]) or
#                      numpy.array_equal(classes, [-1]) or
#                      numpy.array_equal(classes, [1]))):
#             raise ValueError("Data is not binary and pos_label is not specified")
#         elif pos_label is None:
#             pos_label = 1.
#
#         # make y_true a boolean vector
#         y_true = (y_true == pos_label)
#
#         # sort scores and corresponding truth values
#         desc_score_indices = numpy.argsort(y_score, kind="mergesort")[::-1]
#         y_score = y_score[desc_score_indices]
#         y_true = y_true[desc_score_indices]
#
#         # y_score typically has many tied values. Here we extract
#         # the indices associated with the distinct values. We also
#         # concatenate a value for the end of the curve.
#         distinct_value_indices = numpy.where(numpy.diff(y_score))[0]
#         threshold_idxs = numpy.r_[distinct_value_indices, y_true.size - 1]
#
#         # accumulate the true positives with decreasing threshold
#         tps = self.stable_cumsum(y_true)[threshold_idxs]
#         fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing
#
#         thresholds = y_score[threshold_idxs]
#
#         recall = tps / tps[-1]
#
#         last_ind = tps.searchsorted(tps[-1])
#         sl = slice(last_ind, None, -1)  # [last_ind::-1]
#         recall, fps, tps, thresholds = numpy.r_[recall[sl], 1], numpy.r_[fps[sl], 0], numpy.r_[tps[sl], 0], thresholds[sl]
#
#         cutoff = numpy.argmin(numpy.abs(recall - recall_level))
#
#         return fps[cutoff] / (numpy.sum(numpy.logical_not(y_true)))
#
#     @staticmethod
#     def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
#         """Use high precision for cumsum and check that final value matches sum
#         Parameters
#         ----------
#         arr : array-like
#             To be cumulatively summed as flat
#         rtol : float
#             Relative tolerance, see ``numpy.allclose``
#         atol : float
#             Absolute tolerance, see ``numpy.allclose``
#         """
#         out = numpy.cumsum(arr, dtype=numpy.float64)
#         expected = numpy.sum(arr, dtype=numpy.float64)
#         if not numpy.allclose(out[-1], expected, rtol=rtol, atol=atol):
#             raise RuntimeError('cumsum was found to be unstable: '
#                                'its last element does not correspond to sum')
#         return out



