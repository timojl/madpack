import random
import numpy as np

from os.path import join
from madpack.datasets import DatasetBase


class MNIST(DatasetBase):
    """ Implementation of the MNIST Dataset. """

    repository_files = ['MNIST.tar.gz']

    def __init__(self, split, factor=1, augmentation=None, resize_factor=None, seed=None, ):
        super().__init__()

        self.augmentation = augmentation

        x_filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']
        y_filenames = ['train-labels.idx1-ubyte', 't10k-labels-idx1-ubyte']

        data_x, data_y = [], []
        for x_filename in x_filenames:
            with open(join(self.data_path(), x_filename), 'rb') as f:
                data_x += [np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28) / np.float32(256)]

        for y_filename in y_filenames:
            with open(join(self.data_path(), y_filename), 'rb') as f:
                data_y += [np.frombuffer(f.read(), np.uint8, offset=8)]

        self.data_x = np.concatenate(data_x, axis=0)
        self.data_y = np.concatenate(data_y, axis=0)

        if split == 'train':
            interval = 0, 50000
        elif split == 'val':
            interval = 50000, 60000
        elif split == 'test':
            interval = 60000, len(self.data_x)
        else:
            raise ValueError('Invalid split:' + split)

        if seed is not None:  # shuffle data
            indices = list(range(len(self.data_x)))
            random.seed(seed)
            random.shuffle(indices)
            self.data_x = [self.data_x[i] for i in indices]
            self.data_y = [self.data_y[i] for i in indices]

        # self.data_x = self.data_x[interval[0]:interval[1]]
        # self.data_y = self.data_y[interval[0]:interval[1]]

        # this is just bullshit for testing
        self.data_x *= factor

        if resize_factor is not None:  # reduce according to n_samples
            self.data_x = self.data_x[:int(resize_factor * len(self.data_x))]
            self.data_y = self.data_y[:int(resize_factor * len(self.data_y))]
            
        self.sample_ids = tuple(range(interval[0], interval[1]))

    def __getitem__(self, index):

        img = self.data_x[index]

        if self.augmentation:
            tx, ty = np.random.randint(-3, 4), np.random.randint(0, 4)
            img = np.roll(img, tx, axis=0)
            img = np.roll(img, ty, axis=1)

        return (img,), (self.data_y[index].astype('int'),)

