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
        self.ctc_cost = nn.CTCLoss(blank=pad_num)

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = ops.split(l_border, split_size_or_sections=[4, 1], axis=1)
        f_border_split = f_border
        l_border_norm_split = ops.tile(l_border_norm, multiples=[1, 4, 1, 1])
        l_border_score = ops.tile(l_score, multiples=[1, 4, 1, 1])
        l_border_mask = ops.tile(l_mask, multiples=[1, 4, 1, 1])

        border_diff = l_border_split - f_border_split
        abs_border_diff = ops.abs(border_diff)
        border_sign = (abs_border_diff < 1.0).astype(ms.float32)
        border_sign = ops.stop_gradient(border_sign)

        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                         (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss

        border_loss = ops.sum(border_out_loss * l_border_score * l_border_mask) / \
                      (ops.sum(l_border_score * l_border_mask) + 1e-5)
        return border_loss
    
    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = ops.split(l_direction, split_size_or_sections=[2, 1], axis=1)
        f_direction_split = f_direction
        l_direction_norm_split = ops.tile(l_direction_norm, multiples=[1, 2, 1, 1])
        l_direction_score = ops.tile(l_score, multiples=[1, 2, 1, 1])
        l_direction_mask = ops.tile(l_mask, multiples=[1, 2, 1, 1])

        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = ops.abs(direction_diff)
        direction_sign = (abs_direction_diff < 1.0).astype(ms.float32)
        direction_sign = ops.stop_gradient(direction_sign)

        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + \
                            (abs_direction_diff - 0.5) * (1.0 - direction_sign)
        direction_out_loss = l_direction_norm_split * direction_in_loss

        direction_loss = ops.sum(direction_out_loss * l_direction_score * l_direction_mask) / \
                         (ops.sum(l_direction_score * l_direction_mask) + 1e-5)
        return direction_loss
    
    def ctc_loss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = ops.transpose(f_char, (0, 2, 3, 1))
        tcl_pos = ops.reshape(tcl_pos, (-1, 3)).astype(ms.int32)
        f_tcl_char = ops.gather_nd(f_char, tcl_pos)
        f_tcl_char = ops.reshape(f_tcl_char, (-1, 64, self.pad_num + 1))

        f_tcl_char_fg, f_tcl_char_bg = ops.split(f_tcl_char, split_size_or_sections=[self.pad_num, 1], axis=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0

        tcl_mask_fg = ops.tile(tcl_mask, multiples=[1, 1, self.pad_num])
        tcl_mask_fg = ops.stop_gradient(tcl_mask_fg)
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1.0 - tcl_mask_fg) * -20.0

        f_tcl_char_mask = ops.concat((f_tcl_char_fg, f_tcl_char_bg), axis=2)
        f_tcl_char_ld = ops.transpose(f_tcl_char_mask, (1, 0, 2))
        N, B, _ = f_tcl_char_ld.shape
        input_lengths = ops.ones((B,), dtype=ms.int32) * N
        cost = self.ctc_cost(f_tcl_char_ld, tcl_label, input_lengths, label_t)
        return cost
    
    def construct(self, predicts, *labels):
        tcl_maps, tcl_labels_maps, border_maps, direction_maps, \
            training_masks, label_list, pos_list, pos_mask = labels
        