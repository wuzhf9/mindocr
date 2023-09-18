import math
from mindspore import nn, ops

from ._registry import register_backbone, register_backbone_class

__all__ = ["Rec_DenseNet", "rec_densenet"]


class Bottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, pad_mode="pad", padding=1)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), axis=1)
        return out
    

class SingleLayer(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, pad_mode="pad", padding=1)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = self.conv1(ops.relu(x))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), axis=1)
        return out
    

class Transition(nn.Cell):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, pad_mode="pad")
        self.use_drouput = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        if self.use_drouput:
            out = self.dropout(out)
        out = ops.avg_pool2d(out, kernel_size=2, stride=2, ceil_mode=True)
        return out
    
@register_backbone_class
class Rec_DenseNet(nn.Cell):
    def __init__(self, inChannels, growthRate, reduction, use_bottleneck=True, use_dropout=True):
        super(Rec_DenseNet, self).__init__()
        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.out_channels = [684]
        self.conv1 = nn.Conv2d(inChannels, nChannels, kernel_size=7, pad_mode="pad", padding=3, stride=2)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, use_bottleneck, use_dropout):
        layers = []
        for _ in range(nDenseBlocks):
            if use_bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.conv1(x)
        out = ops.relu(out)
        out = ops.max_pool2d(out, kernel_size=2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out


@register_backbone
def rec_densenet(pretrained: bool = False, **kwargs):
    model = Rec_DenseNet(inChannels=1, growthRate=24, reduction=0.5, use_bottleneck=True, use_dropout=True)

    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `rec_densenet` backbone does not exist.")
    
    return model
