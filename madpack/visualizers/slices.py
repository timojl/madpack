from madpack.transforms.spatial import resize
import numpy as np
from functools import partial
from madpack.visualizers.base import VisualizerBase

from madpack import log
# from madpack.transforms import tensor_resize, images_to_grid
from madpack.transforms import images_to_grid

class Slices(VisualizerBase):
    """
    `maps_dim` indicates the dimension along which the maps are stored.
    `maps_dim` and `channel_dim` ignore the batch dimension
    """
    def __init__(self, data_item, target_size, maps_dim, slice_labels=None, # maps_dim=None,
                 channel_dim=None, color=False, normalize=False):
        super().__init__(data_item, target_size)

        self.normalize = normalize
        assert maps_dim is not None, 'maps_dim can not be None'
        # assert channel_dim is not None and color or channel_dim is None

        self.cursor = 0
        self.maps_dim = maps_dim
        self.slice_labels = slice_labels
        if slice_labels is not None:
            log.detail('Using slice', len(slice_labels), 'labels:', slice_labels)
        self.frame = None
        self.channel_dim = channel_dim

        self.target_size = target_size
        self.generate(self.target_size)
        # self.draw_slice()

    def plot(self, ax):

        img = self.as_image()
        norm = 255 if 1 < img.max() <= 255 else 1
        img = img / norm

        if img.ndim == 3:
            img = img.transpose([1, 2, 0])
 
        import matplotlib.pyplot as plt
        ax.axis('off')
        ax.imshow(img, cmap=plt.cm.gray)

    def generate(self, target_size):
        """ Resizes to `target size` but also sets channels last and maps first. """
        # treat the images as channels

        transformed_data = []
        for i in range(self.data_item.shape[self.maps_dim]):
            slice_map = np.take(self.data_item, i, self.maps_dim)

            # ipdb.set_trace()
            if self.channel_dim is not None:
                corrected_channel_dim = self.channel_dim - (1 if self.channel_dim > self.maps_dim else 0)
            else:
                corrected_channel_dim = None

            # resized_map = tensor_resize(slice_map, target_size, interpret_as_max_bound=True,
            #                             channel_dim=corrected_channel_dim, keep_channels_last=True)
            resized_map = resize(slice_map, target_size, channel_dim=corrected_channel_dim)

            if corrected_channel_dim == 0:
                resized_map = resized_map.transpose([1,2,0])
            elif corrected_channel_dim is None:
                resized_map = resized_map[:,:,None]

            transformed_data += [resized_map]

        self.transformed_data = np.array(transformed_data)

        if self.normalize:
            self.transformed_data -= self.transformed_data.min()
            self.transformed_data /= self.transformed_data.max()

    def as_image(self):

        grid_image = images_to_grid(self.transformed_data, self.target_size).numpy()

        if len(grid_image.shape) == 3:
            grid_image = grid_image.transpose([2, 0, 1])

        if grid_image.max() <= 1:
            grid_image *= 255

        return grid_image

    def get_visdom_data(self):
        return self.as_image(), 'image'


DenseM = partial(Slices, maps_dim=0, color=False)
Video = partial(Slices, channel_dim=0, maps_dim=1, color=True)
Video_maps_first = partial(Slices, channel_dim=1, maps_dim=0, color=True)
VideoGS = partial(Slices, channel_dim=0, maps_dim=1, color=False)