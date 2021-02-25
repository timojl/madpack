from functools import partial
from matplotlib import pyplot as plt
import numpy as np

from madpack.visualizers.base import VisualizerBase
from madpack.transforms import resize


class Image(VisualizerBase):
    def __init__(self, data_item, target_size, color=True, channel_dim=0, bgr=False, colormap=None):
        # Frame.__init__(self, parent.mainframe)
        super().__init__(data_item, target_size)

        assert not (color and (colormap is not None)), 'A color image can not be color-mapped'

        self.color = color
        self.channel_dim = channel_dim
        self.bgr = bgr

        assert type(data_item) == np.ndarray

        self.data_item = data_item
        self.generate(target_size)

    def generate(self, target_size=None):

        data_item_view = self.data_item.copy()

        assert data_item_view.ndim in {3, 2}, 'Image must have 3 (color) or 2 (grayscale) dimensions.'

        if self.color:
            if self.channel_dim == 0:
                data_item_view = data_item_view.transpose([1, 2, 0])
            elif self.channel_dim == 1:
                data_item_view = data_item_view.transpose([0, 2, 1])
            elif self.channel_dim == 2:
                data_item_view = data_item_view.transpose([0, 1, 2])

        # correct for negative values
        if -1 <= data_item_view.min() < 0 <= data_item_view.max() <= 1:
            data_item_view += 1
            data_item_view /= 2
        elif -100 <= data_item_view.min() < 0 <= data_item_view.max() <= 100:
            data_item_view -= data_item_view.min()
            data_item_view /= data_item_view.max()
        elif -255 <= data_item_view.min() < 0 <= data_item_view.max() <= 255:
            data_item_view += 255
            data_item_view /= 2

        if self.bgr:
            data_item_view = data_item_view[:, :, [2, 1, 0]]

        if target_size is not None:
            data_item_view = resize(data_item_view, target_size, max_bound=True, channel_dim=2 if len(data_item_view.shape)==3 else 0)
        self.data_transformed = data_item_view
        # return data_item_view

    def as_image(self):
        data_item_view = self.data_transformed.copy()

        if data_item_view.max() <= 1 and data_item_view.dtype.name.startswith('float'):
            data_item_view = np.uint8(data_item_view * 255)
        else:
            data_item_view = np.uint8(data_item_view)

        return data_item_view

    def plot(self, ax):

        if np.issubdtype(self.data_transformed.dtype, np.floating):
            if self.data_transformed.max() > 1.001:
                self.data_transformed = self.data_transformed / 255

        ax.imshow(self.data_transformed, cmap=None if self.color else plt.get_cmap('gray'))
        ax.axis('off')


ImageGrayscale = partial(Image, color=False)
