import mindspore as ms
from mindspore import ops, nn
from mindspore.nn.loss.loss import LossBase

from .det_loss import DiceLoss

__all__ = ["PGLoss"]


class PGLoss(LossBase):
    def __init__(
        self, pad_num, eps=1e-6, **kwargs
    ):
        super(PGLoss, self).__init__()
        self.pad_num = pad_num
        self.dice_loss = DiceLoss(eps=eps)
        self.ctc_cost = nn.CTCLoss(blank=pad_num, reduction="none")

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = ops.split(l_border, split_size_or_sections=[4, 1], axis=1)

        border_diff = l_border_split - f_border
        abs_border_diff = ops.abs(border_diff)
        border_sign = (abs_border_diff < 1.0).astype(ms.float32)
        border_sign = ops.stop_gradient(border_sign)

        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                         (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm * border_in_loss

        border_loss = ops.sum(border_out_loss * l_score * l_mask) / \
                      (ops.sum(l_score * l_mask) + 1e-5)
        return border_loss
    
    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = ops.split(l_direction, split_size_or_sections=[2, 1], axis=1)

        direction_diff = l_direction_split - f_direction
        abs_direction_diff = ops.abs(direction_diff)
        direction_sign = (abs_direction_diff < 1.0).astype(ms.float32)
        direction_sign = ops.stop_gradient(direction_sign)

        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + \
                            (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        direction_out_loss = l_direction_norm * direction_in_loss

        direction_loss = ops.sum(direction_out_loss * l_score * l_mask) / \
                         (ops.sum(l_score * l_mask) + 1e-5)
        return direction_loss
    
    def ctc_loss(self, f_char, tcl_pos, tcl_mask, tcl_label, ctc_mask, label_len):
        f_char = ops.transpose(f_char, (0, 2, 3, 1)) # [bs, 37, 128, 128] -> [bs, 128, 128, 37]
        tcl_pos = ops.reshape(tcl_pos, (-1, 3)).astype(ms.int32) # [bs, 30, 64, 3] -> [bs * 30 * 64, 3]
        f_tcl_char = ops.gather_nd(f_char, tcl_pos) # [bs * 30 * 64, 37]
        f_tcl_char = ops.reshape(f_tcl_char, (-1, 64, self.pad_num + 1)) # [bs * 30, 64, 37]

        f_tcl_char_fg, f_tcl_char_bg = ops.split(f_tcl_char, split_size_or_sections=[self.pad_num, 1], axis=2) # [bs * 30, 64, 36], [bs * 30, 64, 1]
        b, n, l, c = tcl_mask.shape
        tcl_mask = ops.reshape(tcl_mask, (b * n, l, c))
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0

        tcl_mask_fg = ops.tile(tcl_mask, multiples=(1, 1, self.pad_num))
        tcl_mask_fg = ops.stop_gradient(tcl_mask_fg)
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * -20.0

        f_tcl_char_mask = ops.concat((f_tcl_char_fg, f_tcl_char_bg), axis=2) # [bs * 30, 64, 37]
        f_tcl_char_ld = ops.transpose(f_tcl_char_mask, (1, 0, 2)) # [64, bs * 30, 37]
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = ops.ones((B,), dtype=ms.int32) * N

        L = tcl_label.shape[2]
        tcl_label = ops.reshape(tcl_label, (B, L))
        ctc_mask = ops.reshape(ctc_mask, (B,))
        label_len = ops.reshape(label_len, (B,))
        cost = self.ctc_cost(f_tcl_char_ld, tcl_label, input_lengths, label_len)
        mean_cost = ops.sum(cost * ctc_mask) / (ops.sum(ctc_mask) + 1e-5)
        return mean_cost
    
    def construct(self, predicts, *labels):
        tcl_maps, border_maps, direction_maps, training_masks, \
            label_list, pos_list, pos_mask, ctc_mask, label_len = labels
        
        f_score = predicts[0]
        f_border = predicts[1]
        f_char = predicts[2]
        f_direction = predicts[3]

        score_loss = self.dice_loss(f_score, tcl_maps, training_masks)
        border_loss = self.border_loss(f_border, border_maps, tcl_maps, training_masks)
        direction_loss = self.direction_loss(f_direction, direction_maps, tcl_maps, training_masks)
        ctc_loss = self.ctc_loss(f_char, pos_list, pos_mask, label_list, ctc_mask, label_len)
        loss = score_loss + border_loss + direction_loss + 5 * ctc_loss

        return loss
        