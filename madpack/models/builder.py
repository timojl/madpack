import torch
import torch.nn.functional as nnf
from torch import nn


class BlockND(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool=1, padding=(0,0), 
                 stride=1, norm='batch_norm', nonlin='relu', pool_type='avg'):
        super().__init__()
        
        self.padding = padding
        self.nonlins = dict(relu=nn.ReLU, leaky_relu=nn.LeakyReLU)
        norm_layer, conv_bias = self.build_norm(norm, out_channels) 
        
        self.layers = nn.Sequential(
            self.conv_type(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=conv_bias),
            norm_layer,
            self.nonlins[nonlin]() if nonlin is not None else nn.Identity(),
            pool_type(kernel_size=pool) if pool > 1 else nn.Identity()
        )
        
    def build_norm(self, norm_type, channels):
        if norm_type is None:
            return nn.Identity(), True
        if norm_type == 'batch_norm':
            return self.batchnorm_type(channels), False

    def forward(self, x):
        x = nnf.pad(x, self.padding)
        x = self.layers(x)
        return x

    
class Block2D(BlockND):
    
    pool_types = dict(avg=nn.AvgPool2d, max=nn.MaxPool2d)
    conv_type = nn.Conv2d
    batchnorm_type = nn.BatchNorm2d
    
    def __init__(self, *args, padding=(0,0), **kwargs):
        super().__init__(*args, padding=padding, **kwargs)

    
class Block3D(BlockND):
    
    pool_types = dict(avg=nn.AvgPool3d, max=nn.MaxPool3d)
    conv_type = nn.Conv3d
    batchnorm_type = nn.BatchNorm3d
    
    def __init__(self, *args,  padding=(0,0,0,0,0,0), **kwargs):
        super().__init__(*args, padding=padding, **kwargs)
