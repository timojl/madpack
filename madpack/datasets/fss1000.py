import numpy as np
import os
from os.path import join
from itertools import combinations

import torch
from madpack.datasets.base import DatasetBase
from madpack.transforms import imread, resize, pad_to_square, random_crop_slices
from torchvision.transforms import functional as transforms
from madpack.transforms.pipelines import dual_augment_img_with_mask

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class FSS1000(DatasetBase):
    """ FSS1000 is a few-shot segmentation dataset. Here we implementa one-shot setting only. """

    repository_files = ['FSS1000.tar']

    def __init__(self, split, resolution=(96, 96), scale=1, normalize=False, ratio_negative=0,
                 swap=True, with_class=False, aug=0, no_support=False):
        """ Initializes the FSS1000 Dataset
        
        Arguments:
        split: split
        resolution: resolution of query and support image (height=width)
        """
        super().__init__()
        self.resolution = resolution
        self.split = split
        self.no_support = no_support
        self.scale = scale
        self.normalize = normalize
        self.swap = swap
        self.aug = aug
        self.with_class = with_class

        all_classes = sorted(os.listdir(join(self.data_path(), 'classes')))
        test_classes = open(join(self.data_path(), 'fss1000_test_set.txt')).read().split('\n')
        trainval_classes = sorted(list(set(all_classes).difference(test_classes)))
        start, end = 0, None

        if split in {'train', 'val'}:
            classes = trainval_classes[5:] if split == 'train' else trainval_classes[:5]
            # print(classes)
            # start, end = (1, None) if split == 'train' else (0, 1)
        elif split == 'test':
            classes = test_classes
            # start, end = 0, None
        else:
            raise ValueError(f'Invalid split {split}')

        ignore = {('peregine_falcon', 7)}

        self.classes = classes

        samples = []
        for c in classes:
            for s, q in list(combinations(range(10), 2))[start: end]:
                if (c, q) not in ignore and (c, s) not in ignore:
                    samples += [(c, s, c, q)]
                    if swap:
                        samples += [(c, q, c, s)]

        if ratio_negative > 0:
            import random
            n_negatives = int(len(samples) * ratio_negative)
            indices = range(len(samples))

            samples_neg = []
            for _ in range(n_negatives):
                i = random.choice(indices)
                c1, s, c2, q = samples[i]

                while True:
                    c2_new = random.choice(classes)
                    if c2_new != c2:
                        break

                samples_neg += [(c1, s, c2_new, q)]
            
            samples += samples_neg

        self.samples = samples
        self.sample_ids = list(range(len(samples)))

    def __getitem__(self, idx):

        c_s, idx_s, c_q, idx_q = self.samples[idx]
        img_q = imread(join(self.data_path(), 'classes', f'{c_s}/{idx_s+1}.jpg'))

        if c_s == c_q:
            label_q = imread(join(self.data_path(), 'classes', f'{c_s}/{idx_s+1}.png'), grayscale=True)
        else:
            label_q = torch.zeros((1,) + img_q.shape[1:], dtype=torch.float32)

        img_s = imread(join(self.data_path(), 'classes', f'{c_q}/{idx_q+1}.jpg'))
        label_s = imread(join(self.data_path(), 'classes', f'{c_q}/{idx_q+1}.png'), grayscale=True)

        assert all(t is not None for t in [img_q, label_q, img_s, label_s])

        if self.split == 'train':
            image_size = self.resolution[0]+20
        else:
            image_size = self.resolution[0]

        img_q = resize(img_q, (image_size, image_size))
        label_q = resize(label_q, (image_size, image_size), interpolation='nearest')

        if self.split == 'train':
            slices_y, slices_x = random_crop_slices((image_size, image_size), (self.resolution[0], self.resolution[0]))
            img_q = img_q[:, slices_y, slices_x]
            label_q = label_q[:, slices_y, slices_x]

        img_s = resize(img_s, (self.resolution[1], self.resolution[1]))
        img_s = pad_to_square(img_s, channel_dim=0)

        label_s = resize(label_s, (self.resolution[1], self.resolution[1]), interpolation='nearest')

        assert label_s.shape == (1, self.resolution[1], self.resolution[1]), idx

        if self.aug:
            img_q, label_q = dual_augment_img_with_mask(img_q, label_q, from_numpy=False, imagenet_normalize=False, strength=self.aug)
            img_s, label_s = dual_augment_img_with_mask(img_s, label_s, from_numpy=False, imagenet_normalize=False, strength=self.aug)

        if self.normalize:
            img_q = transforms.normalize(img_q, mean=torch.tensor([0.485, 0.456, 0.406]), 
                                         std=torch.tensor([0.229, 0.224, 0.225]))
            img_s = transforms.normalize(img_s, mean=torch.tensor([0.485, 0.456, 0.406]), 
                                         std=torch.tensor([0.229, 0.224, 0.225]))                                         
            # img_q = img_q - IMAGENET_MEAN[:, None, None]
            # img_q = img_q / IMAGENET_STD[:, None, None]
            # img_s = img_s - IMAGENET_MEAN[:, None, None]
            # img_s = img_s / IMAGENET_STD[:, None, None]

        if self.no_support:
            img_s = img_s * 0
            label_s = label_s * 0

        label_s = label_s.squeeze(0)
        img_s_masked = img_s.float(), label_s.byte()

        y_out = [label_q.float()]
        if self.with_class:
            # None because second y-argument is expected to be mask
            y_out += [torch.zeros(0), self.classes.index(c_s)]

        return (img_q.float()*self.scale, *img_s_masked), tuple(y_out)
