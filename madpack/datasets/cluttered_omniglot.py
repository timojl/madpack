# from ava.core.dataset import DatasetBase
from madpack.datasets import DatasetBase

from os.path import join, isdir, dirname
# from ava.core.plots import plot_data
from PIL import Image
import numpy as np


class ClutteredOmniglot(DatasetBase):

    def __init__(self, split, labels='pos', k=4, remove_color=None, dp='new', small=False, target_color=False, 
                 n_samples=1000000, seed=None):

        # repository file depends on arguments
        self.repository_files = {
            (32, False): [('ClutteredOmniglot/k32.tar', 'k32.tar')],
            (64, False): [('ClutteredOmniglot/k64.tar', 'k64')],
            (32, True): [('ClutteredOmniglot/k32-small.tar', 'k32-small.tar')]
        }[k, small]

        super().__init__()
        self.split = split
        self.labels = labels
        self.k = k
        self.dp = dp
        self.remove_color = remove_color
        self.hdf5_handle = None
        self.target_color = target_color
        self.small = small

        if dp == 'old':
            self.data_path = join(self.data_path(), 'data')
            self.sample_ids = list(range({'train': 200000, 'val': 1000, 'test': 500}[split]))
        elif dp == 'hdf':
            import h5py
            self.data_path = join(self.data_path(), f'cluttered_omniglot_{k}.hdf5')
            self.hdf5_handle = h5py.File(self.data_path, 'r')
            self.sample_ids = list(range({'train': 1000000, 'val': 10000, 'test': 10000}[split]))
        else:
            self.data_path = join(self.data_path(), f'k{k}-{n_samples}' + ('-small' if small else ''))
            self.sample_ids = list(range({'train': n_samples, 'val': n_samples // 20, 'test': n_samples // 10}[split]))

            # ugly hack
            if not small and split in {'val', 'test'}:
                self.sample_ids = list(range(10000))

        self.default_loss = 'map_loss'

    def __getitem__(self, idx):

        idx_str = str(idx).zfill(6)

        if self.hdf5_handle is not None:
            img = self.hdf5_handle['images'][idx]
            seg = self.hdf5_handle['segs'][idx]
            target = self.hdf5_handle['targets'][idx]
        else:
            img = np.array(Image.open(f'{self.data_path}/{self.split}_img_{self.k}_{idx_str}.png'))[:, :, :3]
            seg = np.array(Image.open(f'{self.data_path}/{self.split}_seg_{self.k}_{idx_str}.png'))

            if seg.ndim == 3 and seg.shape[2] == 4:
                seg = seg[:, :, :3]

            target = np.array(Image.open(f'{self.data_path}/{self.split}_target_{self.k}_{idx_str}.png'))[:, :, :3]

        img = img.transpose([2, 0, 1]).astype('float32') / 255.0
        target = target.transpose([2, 0, 1]).astype('float32')

        if self.remove_color == 'target':
            target = target.mean(0)
            target *= 1/target.max()
            # target = (target[0] > 0.5).astype('float32')
            inp = img, np.expand_dims(target, 0)
        else:
            inp = img, target

        assert target.sum() > 0, f'failed for index {idx}'

        inp = (inp + (img[:, seg > 100].mean(1),)) if self.target_color else inp

        if self.labels in {'pos', 'pos_map', 'pos_map_label'}:
            a = np.argwhere(seg[:, :] > 100)
            assert len(a) > 0, idx
            pos = a.mean(0)

        if self.labels == 'pos':
            return inp, (pos.astype('float32'),)
        elif self.labels == 'pos_map':
            map_size = 96 if not self.small else 48
            pos_map = np.zeros((map_size, map_size), dtype='float32')
            size = 3
            pos_map[max(0, int(pos[0]) - size): min(map_size, int(pos[0]) + size + 1), max(0, int(pos[1]) - size): min(map_size, int(pos[1]) + size + 1)] = 1
            pos_map[int(pos[0]), int(pos[1])] = 3
            pos_map = pos_map / pos_map.sum()

            pos_map = pos_map[None]

            return inp, (pos_map,)
        elif self.labels == 'pos_map_label':
            label = int(pos[0] * 96) + int(pos[1])
            return inp, (label,)
        if self.labels == 'seg':
            seg = (seg[None, :, :, 0] > 100).astype('float32')

            return inp, (seg,)
        else:
            return inp, (seg,)


class ClutteredOmniglotLoc(ClutteredOmniglot):

    def __init__(self, split, k):
        super().__init__(subset, dp='new', n_samples=2000000, k=k, labels='pos_map', remove_color='target')


class ClutteredOmniglotSeg(ClutteredOmniglot):

    def __init__(self, subset, k):
        super().__init__(subset, dp='new', n_samples=2000000, k=k, labels='seg', remove_color='target')


class ClutteredOmniglotXS(ClutteredOmniglot):

    def __init__(self, subset):
        super().__init__(subset, small=True, dp='new', n_samples=100000, k=32, labels='pos_map', remove_color='target')


