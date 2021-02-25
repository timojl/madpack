from madpack.datasets.base import DatasetBase
from madpack import log
import math
import os

from os.path import join, basename
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, resize, resized_crop
from torchvision import transforms


class DescribableTextures(DatasetBase):
    """ Loads images from the describable texture dataset, ignoring splits """
     
    repository_files = ['describable_textures.tar']
    
    def __init__(self, split, image_size=(128, 128)):
        super().__init__()

        assert split == 'train'

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1)),
            transforms.ToTensor()
        ])

        samples_by_category = dict()
        for a, _, c in os.walk(join(self.data_path(), 'images')):
            samples_by_category[basename(a)] = c

        self.classes = sorted(list(samples_by_category.keys()))

        self.sample_ids = [(name, k) for k, v in samples_by_category.items() for name in v]

    def __getitem__(self, idx):
        name, k = self.sample_ids[idx]
        img = self.transform(Image.open(join(self.data_path(), 'images', k ,name)))
        return (img,), (k,)
