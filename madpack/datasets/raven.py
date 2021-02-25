import numpy as np
import os
from os.path import join, dirname
from random import shuffle, seed as set_seed
from numpy.lib.function_base import diff

from madpack.datasets.base import DatasetBase


class Raven(DatasetBase):

    repository_files = ['Raven/RAVEN-10000-release.zip']

    def __init__(self, split, configuration='all', grayscale=False, seed=0):
        super().__init__()
        self.grayscale = grayscale
        self.split = split

        configurations = os.listdir(self.data_path())

        splits = list(range(0, 10000))

        if seed != 0:
            set_seed(seed)
            shuffle(splits)

        if split == 'train':
            a_range = splits[:6000]  # range(1, 19)
        elif split == 'val':
            a_range = splits[6000:8000]  # range(19, 21)
        elif split == 'test':
            a_range = splits[8000:]  # range(21, 25)
        else:
            raise ValueError(f'Invalid split: {split}')

        self.samples = [(a, conf) for a in a_range for conf in configurations]
        self.sample_ids = list(range(len(self.samples)))

    def __getitem__(self, idx):

        sample_id, config = self.samples[idx]
        
        filename = join(self.data_path(), config, f'RAVEN_{sample_id}_{self.split}.npz')
        sample = np.load(filename)

        panel = sample['image'][:8] / 255.0
        candidates = sample['image'][8:] / 255.0

        solution = int(sample['target'])

        return (panel, candidates), (solution,)
