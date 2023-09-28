from mindspore import nn

from ._registry import register_backbone, register_backbone_class

__all__ = ["E2EResNet50", "e2e_resnet50"]

class ConvNormLayer(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        act=False
    ):
        super(ConvNormLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=(kernel_size - 1) // 2
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


class BottleneckBlock(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True
    ):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=True
        )
        self.conv1 = ConvNormLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=True
        )
        self.conv2 = ConvNormLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1
        )

        if not shortcut:
            self.short = ConvNormLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride
            )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv0(x)
        conv1 = self.conv1(out)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = short + conv2
        out = self.relu(out)
        return out
    
@register_backbone_class
class E2EResNet50(nn.Cell):
    def __init__(self, in_channels=3, layers=[3, 4, 6, 3, 3], **kwargs):
        super(E2EResNet50, self).__init__()
        self.out_channels = []
        self.conv1_1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act=True
        )
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="pad", padding=1)

        num_channels = [64, 256, 512, 1024, 2048]
        num_filters = [64, 128, 256, 512, 512]

        stages = []
        for block_id in range(len(layers)):
            blocks = []
            shortcut = False
            for i in range(layers[block_id]):
                block = BottleneckBlock(
                    in_channels=num_channels[block_id] if i == 0 else num_filters[block_id] * 4,
                    out_channels=num_filters[block_id],
                    stride=2 if i == 0 and block_id != 0 else 1,
                    shortcut=shortcut
                )
                shortcut = True
                blocks.append(block)
            stages.append(nn.SequentialCell(blocks))
        self.stages = nn.CellList(stages)

    def construct(self, x):
        outs = [x]
        out = self.conv1_1(x)
        outs.append(out)
        out = self.pool2d_max(out)

        for block in self.stages:
            out = block(out)
            outs.append(out)
        return outs
    

@register_backbone
def e2e_resnet50(pretrained: bool = False, **kwargs):
    model = E2EResNet50(in_channels=3, layers=[3, 4, 6, 3, 3], **kwargs)

    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `e2e_resnet50` backbone does not exist.")
    
    return model
