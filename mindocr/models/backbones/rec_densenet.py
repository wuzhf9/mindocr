import math
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeUniform

from ._registry import register_backbone, register_backbone_class

__all__ = ["Rec_DenseNet", "rec_densenet"]


class Bottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate, dropout=0.0):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, pad_mode="pad", padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = ops.concat((x, out), axis=1)
        return out
    

class SingleLayer(nn.Cell):
    def __init__(self, nChannels, growthRate, dropout=0.0):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, pad_mode="pad", padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv1(self.relu(x))
        out = self.dropout(out)
        out = ops.concat((x, out), axis=1)
        return out
    

class Transition(nn.Cell):
    def __init__(self, nChannels, nOutChannels, dropout=0.0):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, pad_mode="pad")
        self.dropout = nn.Dropout(p=dropout)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode="pad", ceil_mode=True)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.avg_pool(out)
        return out
    
@register_backbone_class
class Rec_DenseNet(nn.Cell):
    def __init__(self, inChannels, growthRate, reduction, use_bottleneck=True, dropout=0.0):
        super(Rec_DenseNet, self).__init__()
        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.out_channels = [684]
        self.conv1 = nn.Conv2d(inChannels, nChannels, kernel_size=7, stride=2, pad_mode="pad", padding=3)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, use_bottleneck, dropout)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="pad", ceil_mode=True)
        self._init_weights()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, use_bottleneck, dropout):
        layers = []
        for _ in range(nDenseBlocks):
            if use_bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, dropout))
            nChannels += growthRate
        return nn.SequentialCell(layers)

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(HeUniform(math.sqrt(5)), cell.weight.shape, cell.weight.dtype)
                )

    def construct(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out


@register_backbone
def rec_densenet(pretrained: bool = False, **kwargs):
    model = Rec_DenseNet(inChannels=1, growthRate=24, reduction=0.5, use_bottleneck=True, dropout=0.2)

    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `rec_densenet` backbone does not exist.")
    
    return model
