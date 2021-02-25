import numpy as np

from os.path import join, isfile

from madpack.transforms import resize, random_crop, imread
from madpack.datasets import DatasetBase
from madpack import log


class CUB200_2011(DatasetBase):

    def __init__(self, split, image_size=352):
        super().__init__()

        self.images = []
        self.labels = []
        self.augmentation = split != 'test'
        self.image_size = image_size

        with open(join(self.data_path(), 'train_test_split.txt'), 'r') as fh:
            lines = fh.read().split('\n')
            splits = [l.split(' ')[1] for l in lines if len(l) > 0]

        with open(join(self.data_path(), 'images.txt'), 'r') as fh:
            lines = fh.read().split('\n')
            images = [l.split(' ')[1] for l in lines if len(l) > 0]

        with open(join(self.data_path(), 'image_class_labels.txt'), 'r') as fh:
            lines = fh.read().split('\n')
            labels = [l.split(' ')[1] for l in lines if len(l) > 0]

        assert len(splits) == len(images) == len(labels)

        log.warning('There is an intersection between CUB200_2011 test and ImageNet training dataset')

        target_split = {
            'train': '1',
            'train_train': '1',
            'train_val': '1',
            'test': '0'
        }[split]

        valid = {
            'train_train': lambda i: i % 20 != 0,
            'train_val': lambda i: i % 20 == 0,
            'train': lambda i: True,
            'test': lambda i: True
        }[split]

        self.sample_ids = tuple((img, label) for i, (img, split_, label) in enumerate(zip(images, splits, labels))
                                if split_ == target_split and valid(i))

        self.default_loss = 'cross_entropy'

    def __getitem__(self, index):
        img_filename, label = self.sample_ids[index]

        img = imread(join(self.data_path(), 'images', img_filename))

        if self.augmentation:
            img_size = int(self.image_size * np.random.uniform(0.9, 1.1))
            img_size = max(self.image_size, img_size + 10)
            img = resize(img, (img_size, img_size), min_bound=True)
            img = random_crop(img, (self.image_size, self.image_size))

        # img = img[:, :, [2, 1, 0]]
        # img = img.transpose([2, 0, 1])
        img = img.float()

        return (img,), (int(label)-1,)

    def install(self):
        from madpack.utils import extract_archive, download_file
        import os

        target = join(self.data_path(), 'cub.tgz')
        if not isfile(target):
            download_file('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz', target)
        extract_archive(target, self.data_path())
        os.unlink(target)
