from tkinter import Label, W

import numpy as np


class VisualizerBase(object):

    def __init__(self, data_item, target_size):
        self.data_transformed = None
        self.data_item = data_item
        self.target_size = target_size

    def generate(self, target_size):
        pass

    def plot(self, ax):
        raise NotImplementedError

    def get_info(self, frame, column):
        label = Label(frame, text='no info available')
        label.grid(column=column + 2, row=3, sticky=(W,))
        return label


def array_info_text(data_item):
    info = 'shape: {}\ndtype: {}\nmin/max: {} {}'
    info = info.format(data_item.shape, data_item.dtype, np.round(data_item.min(), 5), np.round(data_item.max(), 5))
    return info
