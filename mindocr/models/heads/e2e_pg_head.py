import math

from mindspore import nn
from mindspore.common.initializer import initializer, XavierUniform


class ConvNormLayer(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        act=False
    ):
        super(ConvNormLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act_func = nn.ReLU()
        self.act = act

    def construct(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.act_func(out)
        return out
    

class PGHead(nn.Cell):
    def __init__(self, in_channels, character_dict_path, **kwargs):
        super(PGHead, self).__init__()
        with open(character_dict_path, "rb") as file:
            lines = file.readlines()
            character_length = len(lines) + 1

        conv_score = []
        for i in range(3):
            conv = ConvNormLayer(
                in_channels=in_channels if i == 0 else 64,
                out_channels=128 if i == 2 else 64,
                kernel_size=3 if i == 1 else 1,
                padding=1 if i == 1 else 0,
                act=True
            )
            conv_score.append(conv)
        conv_score.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1
            )
        )
        conv_score.append(nn.Sigmoid())
        self.conv_score = nn.SequentialCell(conv_score)

        conv_border = []
        for i in range(3):
            conv = ConvNormLayer(
                in_channels=in_channels if i == 0 else 64,
                out_channels=128 if i == 2 else 64,
                kernel_size=3 if i == 1 else 1,
                padding=1 if i == 1 else 0,
                act=True
            )
            conv_border.append(conv)
        conv_border.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=4,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1
            )
        )
        self.conv_border = nn.SequentialCell(conv_border)

        conv_char = []
        num_in_channels = [in_channels, 128, 128, 256, 256]
        num_out_channels = [128, 128, 256, 256, 256]
        for i in range(5):
            conv = ConvNormLayer(
                in_channels=num_in_channels[i],
                out_channels=num_out_channels[i],
                kernel_size=3 if i % 2 else 1,
                padding=1 if i % 2 else 0,
                act=True
            )
            conv_char.append(conv)
        conv_char.append(
            nn.Conv2d(
                in_channels=256,
                out_channels=character_length,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1
            )
        )
        self.conv_char = nn.SequentialCell(conv_char)

        conv_direct = []
        for i in range(3):
            conv = ConvNormLayer(
                in_channels=in_channels if i == 0 else 64,
                out_channels=128 if i == 2 else 64,
                kernel_size=3 if i == 1 else 1,
                padding=1 if i == 1 else 0,
                act=True
            )
            conv_direct.append(conv)
        conv_direct.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=2,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1
            )
        )
        self.conv_direct = nn.SequentialCell(conv_direct)
        self._init_weights()
        
    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype)
                )

    def construct(self, x, target=None):
        f_score = self.conv_score(x)
        f_border = self.conv_border(x)
        f_char = self.conv_char(x)
        f_direction = self.conv_direct(x)

        return f_score, f_border, f_char, f_direction
