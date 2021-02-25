import numpy as np
from madpack.visualizers.base import VisualizerBase


class DistributionData(VisualizerBase):

    def generate(self, target_size):
        pass

    def __init__(self, data_item, target_size):
        super().__init__(data_item, target_size)

        text = "\ndtype: " + str(data_item.dtype)
        text += "\nsize: " + str(len(data_item))
        text += "\nsum: " + str(data_item.sum())
        text += "\nargmax: " + str(data_item.argmax())
        text += "\nnan/inf: " + str(np.any(np.isnan(data_item)) or np.any(np.isinf(data_item)))
        self.text = text
        assert type(data_item) == np.ndarray

    def as_bar(self):
        return self.data_item

    def plot(self, ax):
        if len(self.data_item) > 200:
            ax.text(0, 0, 'Too large to visualize', fontsize=15)
            ax.axis('off')
            pass
        else:
            ax.bar(np.arange(0, len(self.data_item)), self.data_item, 0.9)
