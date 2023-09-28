from mindspore import nn

from ..backbones.e2e_resnet50 import ConvNormLayer


class DeConvNormLayer(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super(DeConvNormLayer, self).__init__()
        self.deconv = nn.Conv2dTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        out = self.deconv(x)
        out = self.norm(out)
        return out


class PGFPN(nn.Cell):
    def __init__(self, in_channels, **kwargs):
        super(PGFPN, self).__init__()
        self.out_channels = 128
        conv_012 = []
        num_in_channels = [3, 64, 256]
        num_out_channels = [32, 64, 128]
        for i in range(3):
            conv = ConvNormLayer(
                in_channels=num_in_channels[i],
                out_channels=num_out_channels[i],
                kernel_size=3,
                stride=1
            )
            conv_012.append(conv)
        self.conv_012 = nn.CellList(conv_012)

        self.conv_3 = ConvNormLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2
        )

        conv_4567 = []
        num_in_channels = [64, 64, 128, 128]
        num_out_channels = [64, 128, 128, 128]
        num_kernel_sizes = [3, 3, 3, 1]
        num_strides = [1, 2, 1, 1]
        for i in range(2):
            convs = []
            for j in range(2):
                k = i * 2 + j 
                conv = ConvNormLayer(
                    in_channels=num_in_channels[k],
                    out_channels=num_out_channels[k],
                    kernel_size=num_kernel_sizes[k],
                    stride=num_strides[k]
                )
                convs.append(conv)
            conv_4567.append(nn.SequentialCell(convs))
        self.conv_4567 = nn.CellList(conv_4567)

        conv_hs = []
        num_in_channels = [2048, 2048, 1024, 512, 256]
        num_out_channels = [256, 256, 192, 192, 128]
        for i in range(5):
            conv = ConvNormLayer(
                in_channels=num_in_channels[i],
                out_channels=num_out_channels[i],
                kernel_size=1,
                stride=1
            )
            conv_hs.append(conv)
        self.conv_hs = nn.CellList(conv_hs)

        deconvs = []
        for i in range(4):
            deconv = DeConvNormLayer(
                in_channels=num_out_channels[i],
                out_channels=num_out_channels[i+1],
            )
            deconvs.append(deconv)
        self.deconvs = nn.CellList(deconvs)

        conv_gs = []
        for i in range(4):
            conv = ConvNormLayer(
                in_channels=num_out_channels[i+1],
                out_channels=num_out_channels[i+1],
                kernel_size=3,
                stride=1,
                act=True
            )
            conv_gs.append(conv)
        self.conv_gs = nn.CellList(conv_gs)

        self.conv_f = ConvNormLayer(
            in_channels=num_out_channels[4],
            out_channels=num_out_channels[4],
            kernel_size=1,
            stride=1
        )

        self.relu = nn.ReLU()

    def construct(self, x):
        c0, c1, c2, c3, c4, c5, c6 = x
        # FPN down fusion
        f = [c0, c1, c2]
        for i in range(3):
            f[i] = self.conv_012[i](f[i])
        
        down = self.conv_3(f[0])
        for i in range(2):
            down = down + f[i+1]
            down = self.relu(down)
            down = self.conv_4567[i](down)

        # FPN up fusion
        f = [c6, c5, c4, c3, c2]
        for i in range(5):
            f[i] = self.conv_hs[i](f[i])

        for i in range(4):
            up = self.deconvs[i](f[i])
            up = up + f[i+1]
            up = self.relu(up)
            up = self.conv_gs[i](up)
        up = self.conv_f(up)

        out = down + up
        out = self.relu(out)
        return out
