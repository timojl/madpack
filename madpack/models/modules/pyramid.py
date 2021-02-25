import torch
from torch import nn
from torch.nn import functional as nnf


class PyramidModule(nn.Module):

    def __init__(self, in_features=1000, mid_features=1000, out_features=1000):
        super().__init__()

        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_features, mid_features, (1, 1)), nn.BatchNorm2d(mid_features), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_features, mid_features, (1, 1)), nn.BatchNorm2d(mid_features), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_features, mid_features, (1, 1)), nn.BatchNorm2d(mid_features), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_features, mid_features, (1, 1)), nn.BatchNorm2d(mid_features), nn.ReLU())
        ])
        self.conv_end1 = nn.Conv2d(in_features + 4 * mid_features, out_features, (1, 1), bias=False)

    def forward(self, x):
        args = dict(size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        pyramid = [
            x,
            nnf.interpolate(self.pyramid_convs[0](nnf.adaptive_avg_pool2d(x, (1, 1))), **args),
            nnf.interpolate(self.pyramid_convs[1](nnf.adaptive_avg_pool2d(x, (2, 2))), **args),
            nnf.interpolate(self.pyramid_convs[2](nnf.adaptive_avg_pool2d(x, (3, 3))), **args),
            nnf.interpolate(self.pyramid_convs[3](nnf.adaptive_avg_pool2d(x, (6, 6))), **args),
        ]
        x = torch.cat(pyramid, dim=1)
        x = self.conv_end1(x)
        return x
