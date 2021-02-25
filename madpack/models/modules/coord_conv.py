import torch
from torch import nn
from torch.nn import functional as nnf


class CoordConv1d(nn.Module):
    """ A classic convolution augmented with coordinates in the input """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

        self.conv = nn.Conv1d(in_channels + 1, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):

        coords = torch.linspace(0, 1, x.size(2), device=x.device).view(1, 1, -1)
        x = torch.cat([x, coords], dim=1)

        x = self.conv(x)
        return x


class CoordConv2d(nn.Module):
    """ A classic convolution augmented with coordinates in the input """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        bs = x.size(0)

        coords1 = torch.linspace(0, 1, x.size(2), device=x.device).view(1, 1, -1, 1).repeat(bs, 1, 1, x.size(3))
        coords2 = torch.linspace(0, 1, x.size(3), device=x.device).view(1, 1, 1, -1).repeat(bs, 1, x.size(2), 1)

        x = torch.cat([x, coords1, coords2], dim=1)
        x = self.conv(x)

        return x


class CoordConv3d(nn.Module):
    """ A classic convolution augmented with coordinates in the input """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 only_spatial=False):
        super().__init__()
        self.only_spatial = only_spatial

        self.conv = nn.Conv3d(in_channels + (3 if not only_spatial else 2), out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        bs = x.size(0)
        s = x.shape

        coords2 = torch.linspace(0, 1, s[3], device=x.device).view(1, 1, 1, -1, 1).repeat(1, 1, s[2], 1, s[4])
        coords3 = torch.linspace(0, 1, s[3], device=x.device).view(1, 1, 1, 1, -1).repeat(1, 1, s[2], s[3], 1)

        if not self.only_spatial:
            coords1 = torch.linspace(0, 1, s[2], device=x.device).view(1, 1, -1, 1, 1).repeat(1, 1, 1, s[3], s[4])
            coords = torch.cat([coords1, coords2, coords3], dim=1)
        else:
            coords = torch.cat([coords2, coords3], dim=1)

        coords = coords.repeat(bs, 1, 1, 1, 1)

        x = torch.cat([x, coords], dim=1)
        x = self.conv(x)

        return x
