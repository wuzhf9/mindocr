from seqeval.metrics import f1_score, precision_score, recall_score

from ..utils.misc import AllReduce

__all__ = ["VQASerTokenMetric"]

class VQASerTokenMetric:
    def __init__(self, device_num=1, **kwargs):
        self.clear()
        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["precision", "recall", "hmean"]

    def clear(self):
        self.pred_list = []
        self.gt_list = []

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("Length of inputs should be 2")
        post_res, _ = inputs
        preds = post_res["decode_out"]
        labels = post_res["label_decode_out"]
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    def eval(self):
        metrics = {
            "precision": precision_score(self.gt_list, self.pred_list),
            "recall": recall_score(self.gt_list, self.pred_list),
            "hmean": f1_score(self.gt_list, self.pred_list),
        }
        return metrics
