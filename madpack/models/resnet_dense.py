import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from madpack import log


class DecoderBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        x_up = nnf.interpolate(x, size=(skip_x.size(2), skip_x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x_up, skip_x], dim=1)
        x = self.conv(x)
        x = nnf.relu(x)
        return x


class ResNetSegmentation(nn.Module):
    """
    decoder shape defines different factors for the number of maps in the decoder layers.
    Use small_resnet for RN18.
    """

    def __init__(self, base_network, decoder='m', outputs=10, small_resnet=False, inp=3):
        super().__init__()
        self.resnet = base_network

        if hasattr(self.resnet, 'fc'):
            del self.resnet.fc

        if type(decoder) == str:
            self.dec_sizes = {
                'xs': (32, 32, 16, 16, 8),
                's': (256, 128, 64, 16, 16),
                'm': (256, 128, 64, 32, 32),
                'l': (16, 128, 64, 48, 48)
            }[decoder]
        elif type(decoder) in {list, tuple}:
            self.dec_sizes = decoder
        else:
            raise ValueError('invalid decoder configuration')

        if hasattr(self.resnet, 'sizes') and small_resnet:
            self.enc_sizes = self.resnet.sizes[::-1] + tuple([inp])
        elif hasattr(self.resnet, 'sizes') and not small_resnet:
            self.enc_sizes = [x*4 for x in self.resnet.sizes[::-1]] + [inp]
        elif not small_resnet:
            self.enc_sizes = [2048, 1024, 512, 256, inp]
        else:
            self.enc_sizes = [512, 256, 128, 64, inp]

        self.decoder2 = DecoderBlock(self.enc_sizes[0] + self.enc_sizes[1], self.dec_sizes[0])
        self.decoder3 = DecoderBlock(self.dec_sizes[0] + self.enc_sizes[2], self.dec_sizes[1])
        self.decoder4 = DecoderBlock(self.dec_sizes[1] + self.enc_sizes[3], self.dec_sizes[2])
        self.decoder6 = DecoderBlock(self.dec_sizes[2] + self.enc_sizes[4], self.dec_sizes[4])
        self.post_conv = nn.Conv2d(self.dec_sizes[4], outputs, (1, 1))

        self.layer_seq = [['conv1', 'bn1', 'relu', 'maxpool', 'layer1'], ['layer2'], ['layer3'], ['layer4']]

    def forward(self, x):

        x0 = x
        activations = []
        for layers in self.layer_seq:
            for layer in layers:
                x = getattr(self.resnet, layer)(x)

            activations += [x]

        decoder_activations = []
        for i, layer in enumerate(['decoder2', 'decoder3', 'decoder4']):
            x = getattr(self, layer)(x, activations[-i-2])
            decoder_activations += [x]

        x = self.decoder6(x, x0)
        x = self.post_conv(x)

        return x, activations + decoder_activations


def encoder_to_segmenter(base, **kwargs):
    return ResNetSegmentation(base, **kwargs)


class RN18Dense(ResNetSegmentation):

    def __init__(self, pretrained=False, **kwargs):
        super().__init__(resnet18(pretrained=pretrained), **kwargs, small_resnet=True)


class nRN18Dense(ResNetSegmentation):

    def __init__(self, decoder='s', **kwargs):
        from madpack.models.resnet_generalized import RN18Narrow
        super().__init__(RN18Narrow(), decoder=decoder, **kwargs, small_resnet=True)


class RN34Dense(ResNetSegmentation):

    def __init__(self, pretrained=False, **kwargs):
        super().__init__(resnet34(pretrained=pretrained), **kwargs)


class RN50Dense(ResNetSegmentation):

    def __init__(self, pretrained=False, **kwargs):
        super().__init__(resnet50(pretrained=pretrained), **kwargs)


class RN101Dense(ResNetSegmentation):

    def __init__(self, pretrained=False, **kwargs):
        super().__init__(resnet101(pretrained=pretrained), **kwargs)


class RN152Dense(ResNetSegmentation):

    def __init__(self, pretrained=False, **kwargs):
        super().__init__(resnet152(pretrained=pretrained), **kwargs)


# Narrow Dense ResNets

class NarrowRN18Dense(ResNetSegmentation):

    def __init__(self, channels=(16, 32, 64, 128), decoder_shape='s', **kwargs):
        from madpack.models.resnet_generalized import RN18Narrow
        rn = RN18Narrow(channels)

        super().__init__(rn, decoder=decoder_shape, **kwargs, small_resnet=True)


class NarrowRN50Dense(ResNetSegmentation):

    def __init__(self, channels=(16, 32, 64, 128), decoder_shape='s', **kwargs):
        from madpack.models.resnet_generalized import RN50Narrow

        rn = RN50Narrow(channels)
        super().__init__(rn, decoder=decoder_shape, **kwargs, small_resnet=False)
