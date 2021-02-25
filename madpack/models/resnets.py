import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet


class RN18Classifier(ResNet):
    def __init__(self, inp=3, outputs=100, norm_layer=None):
        super().__init__(BasicBlock, [2, 2, 2, 2], input_channels=inp, num_classes=outputs, norm_layer=norm_layer)

    def forward(self, x):
        return super().forward(x),


class RN50Classifier(ResNet):
    def __init__(self, inp=3, outputs=100, norm_layer=None):
        super().__init__(Bottleneck, [3, 4, 6, 3], input_channels=inp, num_classes=outputs, norm_layer=norm_layer)

    def forward(self, x):
        return super().forward(x),
