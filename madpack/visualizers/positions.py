import numpy as np

from madpack.visualizers.base import VisualizerBase


class Positions(VisualizerBase):
    """
    `maps_dim` indicates the dimension along which the maps are stored.
    `maps_dim` and `channel_dim` ignore the batch dimension
    """

    def generate(self, target_size):
        pass

    def __init__(self, data_item, target_size, frame_size=None):
        super().__init__(data_item, target_size)
        self.frame_size = frame_size

    def plot(self, ax):

        img = self.as_image()
        ax.imshow(img, vmin=0, vmax=1)

    def as_image(self):

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap('Set1')
        data_item = np.array(self.data_item).astype('int')

        if len(self.data_item.shape) == 1:
            data_item = data_item.reshape(1, 1, data_item.shape[0])
        elif len(self.data_item.shape) == 2:
            data_item = np.expand_dims(data_item, 1)

        n_symbols, length, pos = data_item.shape

        d0 = np.ceil(np.sqrt(length))
        d1 = (d0-1) if d0 * (d0-1) >= length else d0
        d0, d1 = int(d0), int(d1)

        max_val = data_item.max()

        frame_size = (max_val + 10, max_val + 10) if self.frame_size is None else self.frame_size

        grid_image = np.ones((d0 * frame_size[0], d1 * frame_size[1], 3))
        for i in range(length):
            iy, ix = int(i // d1), int(i % d1)
            off_y, off_x = iy*frame_size[0], ix*frame_size[1]
            view = grid_image[off_y: off_y + frame_size[0], off_x: off_x + frame_size[1]]  # transformed_data_small[i]
            view *= 0
            view[2:-2, 2:-2] *= 0
            for s in range(n_symbols):
                y, x = data_item[s, i]
                y, x = y + 2, x + 2
                view[y-2: y+2, x-2: x+2] += np.array(cmap(s))[:3]

        grid_image = np.minimum(1, grid_image)
        return grid_image

    def get_visdom_data(self):
        return self.as_image(), 'image'
