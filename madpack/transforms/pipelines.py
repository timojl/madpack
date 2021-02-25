import torch
from torchvision.transforms import functional as transforms
from PIL import Image


def dual_augment_img_with_mask_complex(img, seg, strength=0, split='train', image_size=224, input_range=1.0, crop=False,
                                       imagenet_normalize=False, from_numpy=True, rot=False):

    if from_numpy:
        img = torch.from_numpy(img).permute(2, 0, 1)
        seg = torch.from_numpy(seg)

    if strength > 0 and split == 'train':

        if torch.rand(1).item() > 0.5:
            img = transforms.hflip(img)
            seg = transforms.hflip(seg)

            img = transforms.adjust_brightness(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.5, 2).item())
            img = transforms.adjust_contrast(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.5, 2).item())

            img = transforms.adjust_saturation(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.2, 1.8).item())
            img = transforms.adjust_hue(img, torch.clip(torch.randn(1) * 0.02 * strength, -0.1, 0.1).item())

        if torch.rand(1).item() > 0.5:
            fac = 0.9 + (torch.rand(1).item() / 5)
            
            size = int(fac * img.shape[1]), int(fac * img.shape[2])
            img = transforms.resize(img, size)
            seg = transforms.resize(seg.unsqueeze(0), size, interpolation=Image.NEAREST)[0]
            if rot:
                angle = torch.randn(1).item() * 5 * strength
                img = transforms.rotate(img, angle)
                seg = transforms.rotate(seg.unsqueeze(0), angle, resample=Image.NEAREST)[0]

    img = img / input_range

    if image_size != 'original' and not crop:
        fac = image_size / max(img.shape[1:3])
        size = (fac * torch.tensor(img.shape[1:3])).int().tolist()
        img = transforms.resize(img, size)
        seg = transforms.resize(seg.unsqueeze(0), size, interpolation=Image.NEAREST)[0]
    else:
        size = img.shape[1:3]

    if image_size != 'original' and not crop:
        pad = [image_size - img.shape[1], image_size - img.shape[2]]
        pad = [pad[1] // 2, pad[0] // 2, pad[1] - pad[1] // 2, pad[0] - pad[0] // 2, ]
        img = transforms.pad(img, pad)
        seg = transforms.pad(seg, pad)

    if crop:
        if min(img.shape[1:3]) < crop:
            img = transforms.resize(img, crop)
            seg = transforms.resize(seg.unsqueeze(0), crop, interpolation=Image.NEAREST)[0]

        off_x = torch.randint(0, img.shape[2] - crop + 1, (1,))[0]
        off_y = torch.randint(0, img.shape[1] - crop + 1, (1,))[0]
        img = transforms.crop(img, off_y, off_x, crop, crop)
        seg = transforms.crop(seg, off_y, off_x, crop, crop)

    if imagenet_normalize:
        img = transforms.normalize(img, mean=torch.tensor([0.485, 0.456, 0.406]),
                                   std=torch.tensor([0.229, 0.224, 0.225]))
        img = img * input_range

    return img, seg


def dual_augment_img_with_mask(img, seg, strength=0, split='train', input_range=1.0,
                               imagenet_normalize=False, from_numpy=True, rot=True, shear=True):

    if from_numpy:
        img = torch.from_numpy(img).permute(2, 0, 1)
        seg = torch.from_numpy(seg)

    if strength > 0 and split == 'train':

        if torch.rand(1).item() > 0.5:
            img = transforms.hflip(img)
            seg = transforms.hflip(seg)

            img = transforms.adjust_brightness(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.5, 2).item())
            img = transforms.adjust_contrast(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.5, 2).item())

            img = transforms.adjust_saturation(img, torch.clip(1 + torch.randn(1) * 0.1 * strength, 0.2, 1.8).item())
            img = transforms.adjust_hue(img, torch.clip(torch.randn(1) * 0.02 * strength, -0.1, 0.1).item())

        if torch.rand(1).item() > 0.5:

            angle = torch.randn(1).item() * 5 * strength if rot else 0
            
            trans = (torch.rand(2) * 0.1 * strength)
            trans = (trans * torch.tensor(img.shape[1:])).tolist()

            scale_s = 0.15 * strength
            scale = (1 - scale_s) + torch.rand(1) * 2 * scale_s
            
            shear_angle = (torch.randn(2) * 5).tolist() if shear else [0, 0]
            # shear = [0,0]

            img = transforms.affine(img, angle, trans, scale, shear_angle)
            seg = transforms.affine(seg, angle, trans, scale, shear_angle)

    if imagenet_normalize:
        img = img / input_range
        img = transforms.normalize(img, mean=torch.tensor([0.485, 0.456, 0.406]),
                                   std=torch.tensor([0.229, 0.224, 0.225]))
        img = img * input_range

    return img, seg
