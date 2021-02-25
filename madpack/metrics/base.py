
class BaseMetric(object):

    def __init__(self, metric_names, pred_range=None, gt_index=0, pred_index=0, eval_intermediate=True,
                 eval_validation=True):
        self._names = tuple(metric_names)
        self._eval_intermediate = eval_intermediate
        self._eval_validation = eval_validation

        self._pred_range = pred_range
        self._pred_index = pred_index
        self._gt_index = gt_index

        self.predictions = []
        self.ground_truths = []

    def eval_intermediate(self):
        return self._eval_intermediate

    def eval_validation(self):
        return self._eval_validation

    def names(self):
        return self._names

    def add(self, predictions, ground_truth):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def _get_pred_gt(self, predictions, ground_truth):
        pred = predictions[self._pred_index]
        gt = ground_truth[self._gt_index]

        if self._pred_range is not None:
            pred = pred[:, self._pred_range[0]: self._pred_range[1]]

        return pred, gt