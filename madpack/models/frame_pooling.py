import torch
import torch.nn.functional as nnf
from torch import nn
from torchvision.models import resnet18

from madpack import log


def multi_pool(features_pooled):
    """ assumes `features_pooled` to have three dimensions: batch_size, n_images, features """
    features_pooled1 = torch.max(features_pooled, dim=1)[0]  # max over all images per feature
    features_pooled2 = torch.mean(features_pooled, dim=1)  # mean over all images per feature
    features_pooled3 = torch.min(features_pooled, dim=1)[0]  # min over all images per feature
    features_pooled = torch.cat([features_pooled1, features_pooled2, features_pooled3], dim=1)
    return features_pooled


class SingleFrameFeature(nn.Module):

    def __init__(self, outputs, base, dropout=None, ignore_image=False):
        super().__init__()

        self.ignore_image = ignore_image

        self.base_model = base
        self.features = lambda x: self.base_model(x)
        self.feature_size = self.base_model.feature_size()
        self.do_normalization = False

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.classifier = nn.Linear(self.feature_size, outputs)
        self.parameter_variables = (base, dropout)

    def forward(self, *x_vars):

        img = x_vars[0]

        batch_size = img.size(0)

        assert img.shape[2] == 1, 'requires single frames'
        img = img[:, :, 0]

        if not self.ignore_image:
            features = self.features(img)
            features_pooled = nnf.adaptive_avg_pool2d(features, (1, 1))
            features_pooled = features_pooled.view(batch_size, -1)
        else:
            features_pooled = torch.zeros(batch_size, self.feature_size).cuda()

        if self.dropout is not None:
            features_pooled = self.dropout(features_pooled)

        out = self.classifier(features_pooled)
        return out,


class MultiFramePooling(nn.Module):

    def __init__(self, outputs, base, dropout=None, ignore_image=False, vector_input=None, **kwargs):
        super().__init__()

        log.warning('Unused arguments: {}'.format(','.join(kwargs.keys())))

        self.ignore_image = ignore_image
        self.vector_input = vector_input

        self.base_model = base
        self.features = lambda x: self.base_model(x)
        self.feature_size = self.base_model.feature_size()
        self.do_normalization = False  # normalization is already done in the feature extractor

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        lin_inp = self.feature_size * 3 if self.vector_input is None else 3 * self.vector_input
        self.classifier = nn.Linear(lin_inp, outputs)

        self.parameter_variables = (base, dropout)

    def forward(self, *x_vars):

        img = x_vars[0]

        batch_size = img.size(0)
        n_images = img.size(2)
        assert img.shape[2] > 1, 'requires multiple images'

        if self.ignore_image:
            features_pooled = torch.zeros(batch_size, self.feature_size).cuda()
        elif self.vector_input is None:

            img = img.transpose(1, 2)
            img = img.contiguous().view(img.size(0) * img.size(1), *img.shape[2:])

            features = self.features(img)
            features_pooled = nnf.adaptive_avg_pool2d(features, (1, 1))
            features_pooled = multi_pool(features_pooled.view(batch_size, n_images, -1))
        else:
            features_pooled = multi_pool(img)

        if self.dropout is not None:
            features_pooled = self.dropout(features_pooled)

        out = self.classifier(features_pooled)
        return out,


class SingleFrameFeatureRN18(SingleFrameFeature):
    def __init__(self, outputs):
        rn18 = resnet18(pretrained=True)
        super().__init__(outputs, rn18)


class MultiFramePoolingRN18(MultiFramePooling):
    def __init__(self, outputs, **kwargs):
        rn18 = resnet18(pretrained=True)
        super().__init__(outputs, rn18, **kwargs)
