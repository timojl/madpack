import os
import random
import re
from collections import Counter

import numpy as np
from shutil import copy

from os.path import join, expanduser, join

from madpack.datasets.base import DatasetBase
from madpack.config import MADPACK_CONFIG
from madpack.transforms import random_crop, resize, imread
from madpack import log


class ILSVRC2012(DatasetBase):

    def __init__(self, split, image_size=238, shuffle=False, limit_classes_to=None, scale=1):
        self.image_size = image_size

        self.repository_files = [f'ILSVRC2012_{image_size}.tar']

        super().__init__()
        # self.lock = threading.Lock()

        self.scale = scale
        try:
            all_files = open(join(self.data_path(), 'list_of_files.txt')).read().split()
        except FileNotFoundError:
            log.warning('list of files could not be found. create new one.')
            folders = [f for f in os.listdir(self.data_path()) if f[-4:] not in {'.tar', '.txt'}]
            all_files = []
            for folder in folders:
                all_files += [join(folder, file) for file in os.listdir(join(self.data_path(), folder))]

            open(join(self.data_path(), 'list_of_files.txt'), 'w').write('\n'.join(all_files))

        self.classes = sorted(list(set(f[:9] for f in all_files)))

        if split == 'train':
            all_files = [s for s in all_files if hash(s) % 15 < 10]
        elif split == 'val':
            all_files = [s for s in all_files if 10 <= hash(s) % 15 < 12]
        elif split == 'test':
            all_files = [s for s in all_files if 12 <= hash(s) % 15 < 14]

        elif split == 'train+':
            all_files = [s for s in all_files if hash(s) % 15 < 14]
        elif split == 'val+':
            all_files = [s for s in all_files if 14 <= hash(s) % 15]
        elif split == 'test+':
            raise ValueError('subset test+ does not exist')

        if shuffle:
            random.shuffle(all_files)

        self.class_counts = Counter(s[:9] for s in all_files)

        print('class frequencies, min: {}, max: {}'.format(min(self.class_counts.values()), max(self.class_counts.values())))

        some_classes = [sorted(self.class_counts.keys())[i] for i in [0, 1, 2, 3, 50, 100, 200, 300]]
        print(some_classes)
        print([self.class_counts[c] for c in some_classes])

        if limit_classes_to is not None:

            if type(limit_classes_to) == str:
                valid_classes = open(limit_classes_to).read().split('\n')
                self.classes = [c for c in self.classes if c in valid_classes]
                print(f'{len(self.classes)} classes remaining')
                all_files = [f for f in all_files if f[:9] in valid_classes]

            #valid_classes = set(n for n, _ in self.class_counts.most_common(limit_classes_to))
            #all_files = [f for f in all_files if f[:9] in valid_classes]
            # self.classes = [c for c in self.classes if c in valid_classes]

        self.sample_ids = tuple(all_files)

    def data_path(self):
        return join(expanduser(MADPACK_CONFIG['DATASETS_PATH']), f'{self.__class__.__name__}_{self.image_size}')

    def __getitem__(self, index):
        filename = self.sample_ids[index]

        label_name = filename[:filename.index('/')]

        img = imread(join(self.data_path(), filename))

        img = resize(img, (self.image_size, self.image_size), min_bound=True)
        img = random_crop(img, (self.image_size, self.image_size))
        # img = img[:, :, [2, 1, 0]]
        # img = img.transpose([2, 0, 1])

        img = img * self.scale
        label = self.classes.index(label_name)

        return (img,), (label,)

    def install(self):
        raise IOError('The ImageNet dataset must be downloaded manually from image-net.org (after registration)')
