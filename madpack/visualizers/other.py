import numpy as np
from matplotlib import pyplot as plt

from madpack.visualizers.base import VisualizerBase


class FailData(VisualizerBase):

    def __init__(self, data_item, target_size):
        super().__init__(data_item, target_size)

    def plot(self, ax):
        ax.imshow(np.ones((300, 300)) * 0.8, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        ax.text(0, 300, 'Failed to visualize', fontsize=15)
        ax.axis('off')


class TextData(VisualizerBase):

    def __init__(self, data_item, target_size):
        super().__init__(data_item, target_size)

    def plot(self, ax):
        ax.imshow(np.ones((300, 300)) * 0.8, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        ax.text(10, 290, str(self.data_item).replace('\n', ' ')[:30] + ('...' if len(str(self.data_item)) > 30 else ''), fontsize=8)
        ax.axis('off')

    def as_text(self):
        return str(self.data_item)
