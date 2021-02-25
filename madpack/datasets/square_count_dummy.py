from madpack.datasets import DatasetBase
import torch


class SquareCountDummy(DatasetBase):
    """ A simple dataset with a varying number of squares """

    def install(self):
        pass

    def __init__(self, split, img_size=48):
        super().__init__()
        self.img_size = img_size

        self.sample_ids = []
        for _ in range(1000):
            n_shapes = torch.randint(2, 7, (1,)).item()
            this_sample = []
            for _ in range(n_shapes):
                size = torch.randint(6, 15, (1,)).item()
                pos = tuple(torch.randint(0, self.img_size - size, (2,)).tolist())
                this_sample += [(size, pos)]

            self.sample_ids += [tuple(this_sample)]

    def __getitem__(self, idx):
        shapes = self.sample_ids[idx]
        img = torch.zeros(3, self.img_size, self.img_size)

        for size, pos in shapes:
            img[:, pos[0]:pos[0] + size, pos[1]:pos[1] + size] = 1

        return (img,), (len(shapes),)
