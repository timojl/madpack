from functools import partial

import torch
from torch.nn import functional as nnf
from torch import nn

from madpack.models.modules.coord_conv import CoordConv3d
from madpack import log


class SpaceTime(nn.Module):
    """ Loosely following the LTC model by Varol et al. (slightly modified) """

    def __init__(self, outputs, multiclass=None, input_channels=3, dense=1024, size='small', ignore_input=False,
                 dropout=True, post=None, coord_conv=False, **kwargs):
        super().__init__()

        if len(kwargs.keys()) > 0:
            log.warning('Unused arguments: {}'.format(','.join(kwargs.keys())))

        self.ignore_input = ignore_input
        self.post = post

        fs, linear_sizes = {

            'xxs_no_linear': ([input_channels, 16, 16, 32], None),
            'xxs': ([input_channels, 16, 16, 32], [dense]),

            'xs': ([input_channels, 32, 64, 128], [dense, dense]),
            'xs2': ([input_channels, 32, 64, 128, 256, 256], [dense, dense]),
            'xs3': ([input_channels, 32, 64, 128, 256, 512], [dense, dense]),
            'xs4': ([input_channels, 32, 64, 128, 256, 512], [dense, dense]),

            'xs5': ([input_channels, 8, 16, 32, 64, 128], [256]),

            'xs6': ([input_channels, 32, 64, 256, 512, 1024], [dense, dense]),
            'small': ([input_channels, 32, 64, 128, 256, 256], [dense, dense]),
            'original': ([input_channels, 64, 128, 256, 256, 256], [dense, dense]),
            'faster': ([input_channels, 64, 64, 128, 128, 256], [dense, dense]),
        }[size]

        kernels, kernels_depth, pool, pad, pad_depth, avg_pool = {

            'xxs_no_linear': ([3, 3, 3], [1, 3, 3], [1, 1, 2], [1, 1, 1], [1, 1, 1], (None, 1, 1)),
            'xxs': ([3, 3, 3], [1, 3, 3], [1, 1, 2], [1, 1, 1], [1, 1, 1], (None, 1, 1)),

            'xs': ([3, 3, 3], [1, 3, 3], [1, 1, 2], [1, 1, 1], [1, 1, 1], (3, 1, 1)),
            'xs2': ([3, 3, 2, 2, 2], [1, 1, 3, 3, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], (3, 1, 1)),
            'xs3': ([3, 3, 2, 2, 2], [1, 1, 3, 3, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], (3, 1, 1)),
            'xs4': ([5, 3, 2, 2, 2], [1, 1, 3, 3, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], (3, 1, 1)),

            'xs5': ([5, 5, 2, 2, 2], [1, 1, 5, 5, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], (1, 1, 1)),

            'xs6': ([5, 5, 2, 2, 2], [1, 1, 5, 5, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0], (1, 1, 1)),
            'small': ([3, 3, 3, 3, 2], [1, 1, 3, 3, 2], [1, 1, 2, 2, 2], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], (3, 1, 1)),
            'original': ([3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [1, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], (3, 1, 1)),
            'faster': ([3, 3, 3, 3, 3], [1, 3, 3, 3, 1], [1, 1, 2, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], (3, 1, 1))
        }[size]

        Conv3d = CoordConv3d if coord_conv else nn.Conv3d

        def build_block(i):
            return nn.Sequential(
                Conv3d(fs[i - 1], fs[i], kernel_size=(kernels_depth[i-1], kernels[i-1], kernels[i-1]),
                       padding=(pad_depth[i-1], pad[i-1], pad[i-1])),
                nn.ReLU(),
                nn.BatchNorm3d(fs[i]),
                nn.MaxPool3d((pool[i-1], 2, 2), stride=(pool[i-1], 2, 2)))

        n_outputs = outputs * (multiclass if multiclass is not None else 1)

        self.convs = nn.ModuleList([build_block(i) for i in range(1, len(fs))])

        # self.avg_pool = nn.AdaptiveAvgPool3d(avg_pool)
        self.avg_pool = avg_pool

        if linear_sizes is not None:
            linear = []
            last = avg_pool[0] * avg_pool[1] * avg_pool[2] * fs[-1] if None not in self.avg_pool else linear_sizes[0]
            for l in linear_sizes:
                linear += [nn.Linear(last, l)]
                linear += [nn.ReLU()]
                if dropout:
                    linear += [nn.Dropout()]
                last = l

            linear += [nn.Linear(last, n_outputs)]
            self.linear = nn.Sequential(*linear)
        else:
            self.linear = None

    def forward(self, *x_vars):
        x = x_vars[0]

        if self.ignore_input:
            x = x * 0

        for conv in self.convs:
            x = conv(x)

        if self.avg_pool is not None:
            x = nnf.adaptive_avg_pool3d(x, self.avg_pool)

        if self.linear is not None:
            x = x.view(x.size(0), -1)
            x = self.linear(x)

        if self.post == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.post == 'softmax':
            x = torch.softmax(x, dim=1)

        return x,


LTC = partial(SpaceTime, size='original')
