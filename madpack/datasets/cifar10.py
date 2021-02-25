from madpack.datasets import DatasetBase


from torchvision.datasets import CIFAR10 as _CIFAR10

class CIFAR10(DatasetBase):
    """ Leightweigt wrapper around torchvision CIFAR10 """

    repository_files = ['Cifar10.tar']

    def __init__(self, split):
        super().__init__()

        assert split in {'train', 'test'}, 'currently only test and train split are supported'
        self.cifar = _CIFAR10(self.data_path(), train=split!='test')

    def __len__(self):
        return self.cifar.__len__()

    def __getitem__(self, index):
        return self.cifar.__getitem__(index)
