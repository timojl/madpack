import numpy as np
import os
from os.path import join, dirname, isdir
from random import shuffle, seed as set_seed

from madpack.datasets.base import DatasetBase
from madpack.transforms import imread, resize
from torchvision.transforms.functional import rotate
import torch

class CABC(DatasetBase):

    def check_data_integrity(self):
        if self.repository_files[0][1] == 'baseline.tar':
            return isdir(join(self.data_path(), 'baseline-/'))
        if self.repository_files[0][1] == 'ix1.tar':
            return isdir(join(self.data_path(), 'ix1-/'))
        if self.repository_files[0][1] == 'ix2.tar':
            return isdir(join(self.data_path(), 'ix2/'))
        else:
            return False

    def __init__(self, split, difficulty=0, grayscale=False, seed=0, augmentation=None):

        self.repository_files = {
            0: [('CABC/baseline.tar', 'baseline.tar')],
            1: [('CABC/ix1.tar', 'ix1.tar')],
            3: [('CABC/ix2.tar', 'ix2.tar')]
        }[difficulty]

        super().__init__()
        self.grayscale = grayscale
        self.difficulty = difficulty
        self.augmentation = augmentation

        folders = {
            0: ['baseline-/'],
            1: ['ix1-/'],
            3: ['ix2/'],
            'all': ['baseline-/',  'ix1-/', 'ix2/'],
        }[difficulty]
        self.base_paths = [join(self.data_path(), folder) for folder in folders]
        splits = list(range(1, 51))

        if seed != 0:
            set_seed(seed)
            shuffle(splits)

        if split == 'train':
            a_range = splits[0:36]  # range(1, 19)
        elif split == 'val':
            a_range = splits[36:40] # range(19, 21)
        elif split == 'test':
            a_range = splits[40:51] # range(21, 25)
        else:
            raise ValueError(f'Invalid split: {split}')

        self.samples = [(a, b, bp) for a in a_range for b in range(4000) for bp in range(len(self.base_paths))]
        self.labels = [np.load(join(self.base_paths[0], f'metadata/{i}.npy'))[:,[0,2,4]].astype('U13') for i in range(1,51)]

        # self.samples = np.load(join(self.base_paths[0], 'metadata/combined.npy'))[:, [0,2,4]].astype('U13')
        self.sample_ids = list(range(len(self.samples)))

    def __getitem__(self, idx):

        g, sample, bp = self.samples[idx]
        _, _, positive = self.labels[g-1][sample]

        img = imread(join(self.base_paths[0], f'imgs/{g}/sample_{sample}.png'))
        img = resize(img, (150, 150), interpolation='nearest')

        if self.grayscale:
            img = img[:1, :, :]

        if self.augmentation == 'flip':

            if torch.rand(1).item() > 0.5:
                img = img[:, ::-1, :]

            if torch.rand(1).item() > 0.5:
                img = img[:, :, ::-1]         

        elif self.augmentation == 'rotation':
            img = rotate(img, torch.rand(1).item() * 360)

        img = img - 0.5

        return (img,), (np.array([1.0] if positive=='1' else [0.0], dtype='float32'),)
