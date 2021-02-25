import numpy as np
import torch

from madpack.metrics.base import BaseMetric


class Accuracy(BaseMetric):
    """
    Computes accuracy for classification tasks
    """

    def __init__(self, top=None, pred_range=None, gt_index=0, pred_index=0, name='Acc', argmax_on_gt=False):
        self.pred_range = pred_range
        self.pred_index = pred_index
        self.gt_index = gt_index
        self.argmax_on_gt = argmax_on_gt

        top = [1] + top if top is not None else [1]
        assert all(type(t) == int for t in top)
        names = tuple('{}@{}'.format(name, t) for t in top)
        super().__init__(names)

        self.top = top

    def add(self, predictions, ground_truth):

        pred = predictions[self.pred_index]
        gt = ground_truth[self.gt_index]

        if self.pred_range is not None:
            pred = pred[:, self.pred_range[0]: self.pred_range[1]]

        max_top = max(self.top)
        self.predictions += [pred.topk(max_top)[1].detach().cpu().numpy()]

        if not self.argmax_on_gt:
            self.ground_truths += gt.detach().cpu().numpy().tolist()
        else:
            self.ground_truths += [gt.topk(max_top)[1].detach().cpu().numpy()]

    def value(self):

        preds = np.concatenate(self.predictions)

        gts = np.array(self.ground_truths) if not self.argmax_on_gt else np.concatenate(self.ground_truths)
        hits = np.equal(preds, gts[:, None])

        return [np.mean(hits[:, :t].sum(1) > 0) for t in self.top]


class BinaryAccuracy(BaseMetric):

    def __init__(self, threshold=0.0, pred_range=None, gt_index=0, pred_index=0, apply_sigmoid=False,
                 mistakes=(), name='Acc'):

        self.mistakes = mistakes
        metric_names = tuple([name] + ['{}m-acc'.format(m) for m in self.mistakes])
        super().__init__(metric_names, pred_range, gt_index, pred_index)
        self.threshold = threshold
        self.apply_sigmoid = apply_sigmoid

    def add(self, predictions, ground_truth):
        pred, gt = self._get_pred_gt(predictions, ground_truth)

        pred = torch.sigmoid(pred) if self.apply_sigmoid else pred

        self.predictions += [(pred > self.threshold).detach().cpu().numpy()]
        self.ground_truths += [gt.detach().cpu().numpy().tolist()]

    def value(self):
        preds = np.concatenate(self.predictions)
        gts = np.concatenate(self.ground_truths)

        hits = np.equal(preds, gts)

        missed = hits.shape[1] - hits.sum(1)
        misstakes = [(missed <= m).mean() for m in self.mistakes]

        return tuple([hits.mean()] + misstakes)
