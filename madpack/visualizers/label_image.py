import numpy as np

from madpack.visualizers.base import VisualizerBase
from madpack.transforms import resize
from madpack.visualizers.functional import color_label_image


class LabelImage(VisualizerBase):

    def __init__(self, data_item, target_size, maps_dim=None, n_colors=None):
        super().__init__(data_item, target_size)
        self.maps_dim = maps_dim
        self.n_colors = n_colors
        # self.one_hot = one_hot

        self.data_item = data_item
        self.transformed_data = None
        self.target_size = target_size
        self.data_item_view = None

        self.generate(target_size)

    def generate(self, target_size):
        data_item = self.data_item.copy()

        assert type(self.data_item) == np.ndarray
        assert np.issubdtype(self.data_item.dtype, np.integer)

        if data_item.ndim == 3:
            data_item = np.argmax(data_item, axis=0)

        if self.maps_dim is None:
            assert len(data_item.shape) == 2
        else:
            assert len(data_item.shape) == 3 and 0 <= self.maps_dim <= 2

        data_item = data_item.astype('int16')
        self.data_item_view = resize(data_item, target_size,max_bound=True,channel_dim=self.maps_dim, interpolation='nearest')

        if self.maps_dim is not None:
            labels = self.data_item_view.argmax(2)
        else:
            labels = self.data_item_view

        self.data_transformed = color_label_image(labels, self.n_colors)

    def plot(self, ax):

        vmax = 1 if self.data_transformed.max() <= 1 else 255

        ax.imshow(self.data_transformed, vmin=0, vmax=vmax)
        ax.axis('off')
