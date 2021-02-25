from madpack.datasets.base import DatasetBase
from madpack import log
import math
from os.path import join
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, resize


def load_seq(base, k=1):
    images, masks = [], []
    for j in range(15, 35):
        images += [to_tensor(Image.open(join(base, str(k), f'{j}.png')))]
        masks += [to_tensor(Image.open(join(base, str(k), f'm0_00{j}.png')))]
    
    return torch.stack(images), torch.stack(masks)
    

def imagenet_seq(base_img, size=48, min_speed=0):

    target_size = torch.randint(60, 128, (1,)).item()
    # target_size = 128
    base_img = resize(base_img, target_size)
    
    # sample parameters for linear motion
    trials = 0
    while True:  
        off_x, off_y = torch.randint(0, target_size - size, (2,))
        dir_x, dir_y = torch.randint(-5, 5, (2,))
        last = off_x + 10 * dir_x, off_y + 10 * dir_y

        speed = math.sqrt(math.pow(dir_x + dir_y, 2))

        if 0 <= last[0] <= target_size - size and 0 <= last[1] <= target_size - size and speed >= min_speed:
            break

        trials += 1

        if trials > 1000:
            print(target_size, size)
            aaaa
          
    images = []
    for i in range(10):
        ox, oy = off_x + i * dir_x, off_y + i * dir_y
        images += [base_img[:, oy: oy + size, ox: ox + size]]
        
    return torch.stack(images)
    

def overlay(img_a, img_b, mask):
    return img_a * mask + img_b * (1 - mask)


class RotatingObjects(DatasetBase):
     
    repository_files = ['object_centric/rot_obj_33k.tar']
    
    def __init__(self, split, background='imagenet'):
        super().__init__()

        assert split == 'train'

        log.warning('Split is currently ignored')

        if background == 'imagenet':
            from madpack.datasets import ILSVRC2012
            self.imagenet = ILSVRC2012('train', image_size=128)
        elif background == 'texture':
            from madpack.datasets import DescribableTextures
            self.imagenet = DescribableTextures('train', image_size=128)            
        else:
            self.imagenet = None

        self.base = self.data_path()
        self.sample_ids = list(range(33000))

    def __getitem__(self, i):
        images, masks = load_seq(self.base, i)

        if self.imagenet is not None:
            imgnet_idx = torch.randint(0, len(self.imagenet), (1,)).item()
            random_imagenet_image,  = self.imagenet[imgnet_idx][0]
            bg_seq = imagenet_seq(random_imagenet_image, min_speed=1)
        else:
            bg_seq = torch.zeros(10, 3, 48, 48)

        seq, mask_seq = [], []
        for i in range(10):
            seq += [overlay(images[i][:3], bg_seq[i], images[i][3])]
            mask_seq += [masks[i][0]]

        seq = torch.stack(seq).transpose(0, 1)  # channels first
        mask_seq = torch.stack(mask_seq)

        return (seq,), (mask_seq,)
