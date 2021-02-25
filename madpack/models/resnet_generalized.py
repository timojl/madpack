import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet


class ResNetGeneralized(ResNet):
    """ original implementation adapted from torchvision with the following additions

    Args:
      channel_sizes: Controls the networks width and enables building very narrow ResNets
      stride1_layers: Avoids striding and thus reduction of spatial resolution
      maxpool: removes or changes the maxpooling parameters after first convolution.

    Example:
    >> print('Hello World')

    """
    def __init__(self, block, layers, channel_sizes=(64,128,256,512), num_classes=1000, groups=1, input_channels=3,
                 first_kernel=7, width_per_group=64, norm_layer=None, stride1_layers=(), with_fc=True, maxpool=None):
        super().__init__(block, layers, num_classes=num_classes, width_per_group=width_per_group,
                         replace_stride_with_dilation=None, norm_layer=norm_layer)

        self.inplanes = channel_sizes[0]
        self.sizes = channel_sizes
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        overlap = first_kernel - 1
        pad = overlap // 2, overlap - (overlap // 2)

        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=first_kernel, stride=2, padding=pad,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)

        if maxpool == 'remove':
            self.maxpool = nn.Identity()
        elif type(maxpool) in {tuple, list}:
            self.maxpool = nn.MaxPool2d(kernel_size=maxpool[0], stride=maxpool[1], padding=1)
        elif maxpool is None:
            pass 
        else:
            raise ValueError(f'Invalid value for maxpool: {maxpool}')

        self.layer1 = self._make_layer(block, channel_sizes[0], layers[0])
        self.layer2 = self._make_layer(block, channel_sizes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channel_sizes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channel_sizes[3], layers[3], stride=2)

        layer0 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.layers = [layer0, self.layer1, self.layer2, self.layer3, self.layer4]

        if 0 in stride1_layers:
            self.layers[0][0].stride = 1

        for layer_id in set(stride1_layers) - set([0]):
            
            c = self.layers[layer_id][0].conv1
            self.layers[layer_id][0].conv1 = nn.Conv2d(c.in_channels, c.out_channels, kernel_size=c.kernel_size, 
                                                       stride=1, padding=c.padding, bias=False)
            c = self.layers[layer_id][0].conv2
            self.layers[layer_id][0].conv2 = nn.Conv2d(c.in_channels, c.out_channels, kernel_size=c.kernel_size, 
                                                       stride=1, padding=c.padding, bias=False)                                                       

            if self.layers[layer_id][0].downsample is not None:
                d = self.layers[layer_id][0].downsample[0]
                self.layers[layer_id][0].downsample[0] = nn.Conv2d(d.in_channels, d.out_channels, kernel_size=d.kernel_size, 
                                                                   stride=1, padding=d.padding, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        output_size = channel_sizes[3] if block == BasicBlock else 2048

        if with_fc:
            self.fc = nn.Linear(output_size, num_classes)
            self.flatten = True
        else:
            self.avgpool = None
            self.flatten = False
            self.fc = None   

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
    
        if self.avgpool is not None:
            x = self.avgpool(x)
    
        if self.flatten:
            x = x.view(x.size(0), -1)
    
        if self.fc is not None:
            x = self.fc(x)

        return x

class ResNet18Generalized(ResNetGeneralized):

    def __init__(self, channel_sizes=(64,128,256,512), num_classes=1000, groups=1, input_channels=3,
                 first_kernel=7, width_per_group=64, norm_layer=None, stride1_layers=(), with_fc=True, maxpool=None, pretrained=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], channel_sizes=channel_sizes, num_classes=num_classes, groups=groups,
                         input_channels=input_channels, first_kernel=first_kernel, width_per_group=width_per_group, 
                         norm_layer=norm_layer, stride1_layers=stride1_layers, with_fc=with_fc, maxpool=maxpool)

        from torchvision.models.resnet import model_urls
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet18'],  progress=True)
            try:
                self.load_state_dict(state_dict)         
            except RuntimeError as e:
                print(e)
                self.load_state_dict(state_dict, strict=False)         

class ResNet50Generalized(ResNetGeneralized):

    def __init__(self, channel_sizes=(64,128,256,512), num_classes=1000, groups=1, input_channels=3,
                 first_kernel=7, width_per_group=64, norm_layer=None, stride1_layers=(), with_fc=True, maxpool=None, pretrained=False):
        super().__init__(Bottleneck, [3, 4, 6, 3], channel_sizes=channel_sizes, num_classes=num_classes, groups=groups,
                         input_channels=input_channels, first_kernel=first_kernel, width_per_group=width_per_group, 
                         norm_layer=norm_layer, stride1_layers=stride1_layers, with_fc=with_fc, maxpool=maxpool)

        from torchvision.models.resnet import model_urls
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'],  progress=True)
            try:
                self.load_state_dict(state_dict)         
            except RuntimeError as e:
                print(e)
                self.load_state_dict(state_dict, strict=False)           


class RN18ZeroStride_0_2(ResNetGeneralized):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7, norm_layer=None,
                 stride1_layers=(0, 2,), with_fc=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs, norm_layer=norm_layer,
                         stride1_layers=stride1_layers, with_fc=with_fc)

    def forward(self, x):
        return super().forward(x),


class RN18LessPool2(ResNetGeneralized):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7, norm_layer=None,
                 stride1_layers=(0, 2, 3), with_fc=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs, norm_layer=norm_layer,
                         stride1_layers=stride1_layers, with_fc=with_fc)

    def forward(self, x):
        return super().forward(x),


class RN18LessPool3(ResNetGeneralized):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7, norm_layer=None,
                 stride1_layers=(0, 2, 3, 4), with_fc=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs, norm_layer=norm_layer,
                         stride1_layers=stride1_layers, with_fc=with_fc)

    def forward(self, x):
        return super().forward(x),



class RN18Narrow(ResNetGeneralized):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7, norm_layer=None, with_fc=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs, norm_layer=norm_layer, with_fc=with_fc)

    def forward(self, x):
        return super().forward(x),


class RN50Narrow(ResNetGeneralized):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7):
        super().__init__(Bottleneck, [3, 4, 6, 3], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs)

    def forward(self, x):
        return super().forward(x),


# Vanilla ResNets

class RN18Classifier(ResNetGeneralized):
    def __init__(self, inp=3, outputs=100, norm_layer=None):
        super().__init__(BasicBlock, [2, 2, 2, 2], input_channels=inp, num_classes=outputs, norm_layer=norm_layer)

    def forward(self, x):
        return super().forward(x),


class RN50Classifier(ResNetGeneralized):
    def __init__(self, inp=3, outputs=100, norm_layer=None):
        super().__init__(Bottleneck, [3, 4, 6, 3], input_channels=inp, num_classes=outputs, norm_layer=norm_layer)

    def forward(self, x):
        return super().forward(x),

class RN152Classifier(ResNetGeneralized):
    def __init__(self, inp=3, outputs=100, norm_layer=None):
        super().__init__(Bottleneck, [3, 8, 36, 3], input_channels=inp, num_classes=outputs, norm_layer=norm_layer)

    def forward(self, x):
        return super().forward(x),