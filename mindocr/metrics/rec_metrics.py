"""Metric for accuracy evaluation."""
import logging
import re
import numpy as np

from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

import mindspore as ms
from mindspore import nn

from ..utils.misc import AllReduce

__all__ = ["RecMetric", "CANMetric"]
_logger = logging.getLogger(__name__)


class RecMetric(nn.Metric):
    """
    Define accuracy metric for warpctc network.

    Args:
        ignore_space: remove space in prediction and ground truth text if True
        filter_ood: filter out-of-dictionary characters(e.g., '$' for the default digit+en dictionary) in
            ground truth text. Default is True.
        lower: convert GT text to lower case. Recommend to set True if the dictionary does not contains upper letters
        ignore_symbol: Ignore the symbols in the predictions

    Notes:
        Since the OOD characters are skipped during label encoding in data transformation by default,
        filter_ood should be True. (Paddle skipped the OOD character in label encoding and then decoded the label
        indices back to text string, which has no ood character.
    """

    def __init__(
        self,
        character_dict_path=None,
        ignore_space=True,
        filter_ood=True,
        lower=True,
        ignore_symbol=False,
        print_flag=False,
        device_num=1,
        **kwargs,
    ):
        super().__init__()
        self.clear()
        self.ignore_space = ignore_space
        self.filter_ood = filter_ood
        self.lower = lower
        self.ignore_symbol = ignore_symbol
        self.print_flag = print_flag

        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["acc", "norm_edit_distance"]

        if self.ignore_symbol:
            self.valid_symbol = re.compile(r"[^A-Z^a-z^0-9^\u4e00-\u9fa5]")

        # TODO: use parsed dictionary object
        if character_dict_path is None:
            self.dict = [c for c in "0123456789abcdefghijklmnopqrstuvwxyz"]
        else:
            self.dict = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    self.dict.append(c)

    def clear(self):
        self._correct_num = ms.Tensor(0, dtype=ms.int32)
        self._total_num = ms.Tensor(0, dtype=ms.float32)  # avoid int divisor
        self._norm_edit_dis = ms.Tensor(0.0, dtype=ms.float32)

    def update(self, *inputs):
        """
        Updates the internal evaluation result

        Args:
            inputs (tuple): contain two elements preds, gt
                    preds (dict): prediction output by postprocess, keys:
                        - texts, List[str], batch of predicted text strings, shape [BS, ]
                        - confs (optional), List[float], batch of confidence values for the prediction
                    gt (tuple or list): ground truth, order defined by output_columns in eval dataloader.
                        require element:
                        gt_texts, for the grouth truth texts (padded to the fixed length), shape [BS, ]
                        gt_lens (optional), length of original text if padded, shape [BS, ]

        Raises:
            ValueError: If the number of the inputs is not 2.
        """

        if len(inputs) != 2:
            raise ValueError("Length of inputs should be 2")
        preds, gt = inputs
        pred_texts = preds["texts"]
        # pred_confs = preds['confs']

        # remove padded chars in GT
        if isinstance(gt, tuple) or isinstance(gt, list):
            gt_texts = gt[0]  # text string padded
            gt_lens = gt[1]  # text length

            if isinstance(gt_texts, ms.Tensor):
                gt_texts = gt_texts.asnumpy()
                gt_lens = gt_lens.asnumpy()

            gt_texts = [gt_texts[i][:l] for i, l in enumerate(gt_lens)]
        else:
            gt_texts = gt
            if isinstance(gt_texts, ms.Tensor):
                gt_texts = gt_texts.asnumpy()

        for pred, label in zip(pred_texts, gt_texts):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                label = label.replace(" ", "")

            if self.lower:  # convert to lower case
                label = label.lower()
                pred = pred.lower()

            if self.filter_ood:  # filter out of dictionary characters
                label = "".join([c for c in label if c in self.dict])

            # remove symbols
            if self.ignore_symbol:
                label = self.valid_symbol.sub("", label)
                pred = self.valid_symbol.sub("", pred)

            if self.print_flag:
                _logger.info(f"{pred} :: {label}")

            edit_distance = Levenshtein.normalized_distance(pred, label)
            self._norm_edit_dis += edit_distance
            if pred == label:
                self._correct_num += 1

            self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError("Accuary can not be calculated, because the number of samples is 0.")
        _logger.info(f"correct num: {self._correct_num}, total num: {self._total_num}")

        if self.all_reduce:
            # sum over all devices
            correct_num = self.all_reduce(self._correct_num)
            norm_edit_dis = self.all_reduce(self._norm_edit_dis)
            total_num = self.all_reduce(self._total_num)
        else:
            correct_num = self._correct_num
            norm_edit_dis = self._norm_edit_dis
            total_num = self._total_num

        sequence_accurancy = float((correct_num / total_num).asnumpy())
        norm_edit_distance = float((1 - norm_edit_dis / total_num).asnumpy())

        return {"acc": sequence_accurancy, "norm_edit_distance": norm_edit_distance}


class CANMetric(nn.Metric):
    def __init__(self, device_num=1, **kwargs):
        super(CANMetric, self).__init__()
        self.clear()
        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["exp_rate", "word_rate"]

    def clear(self):
        self._word_right = ms.Tensor(0.0, dtype=ms.float32)
        self._exp_right = ms.Tensor(0.0, dtype=ms.float32)
        self._total_length = ms.Tensor(0, dtype=ms.int32)
        self._total_batch = ms.Tensor(0, dtype=ms.int32)

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("Length of inputs should be 2")
        preds, gt = inputs
        preds = preds["preds"]
        labels, labels_mask = gt[0], gt[1]
        if isinstance(labels, ms.Tensor):
            labels = labels.asnumpy()
            labels_mask = labels_mask.asnumpy()
        word_scores = [
            SequenceMatcher(
                None,
                s1[:int(np.sum(s3))],
                s2[:int(np.sum(s3))],
                autojunk=False).ratio() * (
                    len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) /
            len(s1[:int(np.sum(s3))]) / 2
            for s1, s2, s3 in zip(labels, preds, labels_mask)
        ]

        line_right = 0
        batch_size, word_length = labels.shape[:2]
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1

        word_rate = np.mean(word_scores)
        exp_rate = line_right / batch_size

        self._word_right += word_rate * word_length
        self._exp_right += exp_rate * batch_size
        self._total_length += word_length
        self._total_batch += batch_size

    def eval(self):
        if self._total_length == 0:
            raise RuntimeError("Word rate can not be calculated, because the number of words is 0.")
        if self._total_batch == 0:
            raise RuntimeError("Exp rate can not be calculated, because the number of samples is 0.")

        if self.all_reduce:
            # sum over all devices
            word_right = self.all_reduce(self._word_right)
            exp_right = self.all_reduce(self._exp_right)
            total_length = self.all_reduce(self._total_length)
            total_batch = self.all_reduce(self._total_batch)
        else:
            word_right = self._word_right
            exp_right = self._exp_right
            total_length = self._total_length
            total_batch = self._total_batch

        final_word_rate = float((word_right / total_length).asnumpy())
        final_exp_rate = float((exp_right / total_batch).asnumpy())

        return {"ExpRate": final_exp_rate, "WordRate": final_word_rate}


if __name__ == "__main__":
    gt = ["ba xla la!    ", "ba       "]
    gt_len = [len("ba xla la!"), len("ba")]

    pred = ["balala", "ba"]

    m = RecMetric()
    m.update({"texts": pred}, (gt, gt_len))
    acc = m.eval()
    print(acc)
