import numpy as np
import os
from os.path import join, dirname, isdir
from random import shuffle, seed as set_seed
from PIL import Image

from madpack.datasets.base import DatasetBase
from madpack.transforms import resize, imread


class Pathfinder(DatasetBase):

    def check_data_integrity(self):
        if self.repository_files[0][1] == '6.tar':
            return isdir(join(self.data_path(), 'curv_baseline')) and isdir(join(self.data_path(), 'curv_baseline_neg'))
        if self.repository_files[0][1] == '9.tar':
            return isdir(join(self.data_path(), 'curv_contour_length_9')) and isdir(join(self.data_path(), 'curv_contour_length_9_neg'))
        if self.repository_files[0][1] == '14.tar':
            return isdir(join(self.data_path(), 'curv_contour_length_14')) and isdir(join(self.data_path(), 'curv_contour_length_14_neg'))
        else:
            return False

    def __init__(self, split, path_length=6, grayscale=False, seed=0, image_size=(150, 150), augmentation=None):

        self.repository_files = {
            6: [('Pathfinder/6.tar', '6.tar')],
            9: [('Pathfinder/9.tar', '9.tar')],
            14: [('Pathfinder/14.tar', '14.tar')]
        }[path_length]

        super().__init__()
        self.grayscale = grayscale
        self.image_size = image_size
        self.augmentation = augmentation

        folders = {
            6: ['curv_baseline'],
            9: ['curv_contour_length_9'],
            14: ['curv_contour_length_14'],
            'all': ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14'],
        }[path_length]
        self.base_paths = [join(self.data_path(), folder) for folder in folders]
        splits = list(range(1, 25))

        if seed != 0:
            set_seed(seed)
            shuffle(splits)

        if split == 'train':
            a_range = splits[0:18]  # range(1, 19)
        elif split == 'val':
            a_range = splits[18:20]  # range(19, 21)
        elif split == 'test':
            a_range = splits[20:25]  # range(21, 25)
        else:
            raise ValueError(f'Invalid split: {split}')

        self.samples = [(a, b, pos, bp) for a in a_range for b in range(10000) for pos in [True, False] for bp in range(len(self.base_paths))]
        # self.labels = np.load(join(self.base_path, 'metadata', '1.npy'))

        self.sample_ids = list(range(len(self.samples)))

    def __getitem__(self, idx):
        a, b, positive, bp = self.samples[idx]

        suffix = '' if positive else '_neg'
        img = imread(join(self.base_paths[bp] + suffix, f'imgs/{a}/sample_{b}.png'), grayscale=True)
        img = resize(img, self.image_size, interpolation='nearest')

        if self.grayscale:
            img = img[:1, :, :]

        if self.augmentation == 'flip':

            if np.random.random() > 0.5:
                img = img[:, ::-1, :]

            if np.random.random() > 0.5:
                img = img[:, :, ::-1]         

        elif self.augmentation == 'rotation':
            img = Image.fromarray(img)
            img = np.array(img.rotate(np.random.random() * 360, resample=Image.BICUBIC))
            img = np.clip(img, 0, 255)

        img = img / 255
        img = img - 0.5

        return (img,), (np.array([1.0 if positive else 0.0], dtype='float32'),)