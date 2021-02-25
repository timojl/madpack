import torch
import math
import numpy as np


def resize(img, size, interpolation='bilinear', max_bound=False, min_bound=False, channel_dim=0):
    """ convenience wrapper of resize """
    from torchvision.transforms.functional import resize as torch_resize
    from PIL import Image

    assert channel_dim in {0, 2, None}
    assert not min_bound or not max_bound

    to_numpy, drop_first = False, False
    if type(img) == np.ndarray:
        img, to_numpy = torch.tensor(img), True

    if channel_dim == 2:
        img = img.permute(2, 0, 1)

    if len(img.shape) == 2:
        img, drop_first = img.unsqueeze(0), True        

    if min_bound or max_bound:
        factors = size[0] / img.shape[1], size[1] / img.shape[2] 
        if min_bound:
            i = 0 if factors[0] > factors[1] else 1
        else:
            i = 0 if factors[0] < factors[1] else 1
        target_size = [int(img.shape[1] * factors[i]), int(img.shape[2] * factors[i])]
        target_size[i] = size[i]
        size = target_size
    
    interpolations = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
    }
    img_resized = torch_resize(img, size, interpolations[interpolation])

    if drop_first:
        img_resized = img_resized[0]

    if channel_dim == 2:
        img_resized = img_resized.permute(1,2, 0)

    if to_numpy:
        img_resized = img_resized.numpy()

    return img_resized


def pad_to_square(img, channel_dim=2, fill=0):
    """


    add padding such that a squared image is returned """
    
    from torchvision.transforms.functional import pad

    if channel_dim == 2:
        img = img.permute(2, 0, 1)
    elif channel_dim == 0:
        pass
    else:
        raise ValueError('invalid channel_dim')

    h, w = img.shape[1:]
    pady1 = pady2 = padx1 = padx2 = 0

    if h > w:
        padx1 = (h - w) // 2
        padx2 = h - w - padx1
    elif w > h:
        pady1 = (w - h) // 2
        pady2 = w - h - pady1

    img_padded = pad(img, padding=(padx1, pady1, padx2, pady2), padding_mode='constant')

    if channel_dim == 2:
        img_padded = img_padded.permute(1, 2, 0)

    return img_padded


def random_crop_slices(origin_size, target_size):
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}'

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def random_crop(tensor, target_size, spatial_dims=(1, 2)):
    """ Randomly samples a crop of size `target_size` from `tensor` along `image_dimensions` """
    assert len(spatial_dims) == 2 and type(spatial_dims[0]) == int and type(spatial_dims[1]) == int

    # slices = random_crop_slices(tensor, target_size, image_dimensions)
    origin_size = tensor.shape[spatial_dims[0]], tensor.shape[spatial_dims[1]]
    slices_y, slices_x = random_crop_slices(origin_size, target_size)

    slices = [slice(0, None) for _ in range(len(tensor.shape))]
    slices[spatial_dims[0]] = slices_y
    slices[spatial_dims[1]] = slices_x
    slices = tuple(slices)
    return tensor[slices]


def random_crop_special(area, image_size, size, adapt_size=False):
    
    off_x, off_y, width, height = area
    must_contain = torch.zeros(image_size).bool()
    must_contain[off_y: off_y + height, off_x: off_x + width] = 1
    return random_crop_special_by_map(must_contain, size, adapt_size)
    
    
def random_crop_special_by_map(must_contain, size, adapt_size=False, min_overlap=1.0, area=None):

    if type(size) == int:
        size = size, size

    must_contain = must_contain.byte()
    
    if area is not None:
        off_x, off_y, width, height = area
        minx, miny = off_x, off_y
        maxx, maxy = off_x + width, off_y + height    
    else:
        proj_x = must_contain.byte().max(0).values
        proj_y = must_contain.byte().max(1).values
        minx, miny = proj_x.argmax(), proj_y.argmax()
        maxx, maxy = len(proj_x) - torch.flip(proj_x, [0]).argmax(), len(proj_y) - torch.flip(proj_y, [0]).argmax()
        width, height = maxx - minx, maxy - miny
    
    # print(minx, maxx, miny, maxy, (height, width))
    assert not adapt_size or min_overlap == 1.0
    
    # size cannot fit in
    if maxy - miny > size[0] or maxx - minx > size[1]:
        if adapt_size:
            size = max(maxy - miny, size[0]), max(maxx - minx, size[1])
        else:
            raise ValueError('fail')

    target_sum = height * width
    
    iters = 0
    while True:
        iters += 1

        lower_y = max(0, maxy - size[0])
        lower_x = max(0, maxx - size[1])
        
        upper_y = min(miny + 1, must_contain.shape[0] - size[0] + 1)
        upper_x = min(minx + 1, must_contain.shape[1] - size[1] + 1)

        off_y = torch.randint(lower_y, upper_y, (1,)).item()
        off_x = torch.randint(lower_x, upper_x, (1,)).item()
        
        if must_contain[off_y:off_y + size[0], off_x: off_x + size[1]].sum() == target_sum:
            return off_y, off_x, size, iters


def images_to_grid(images, target_size, layout=None, spacing=5, scale_to_fit=True):

    if type(images) == np.ndarray:
        images = torch.from_numpy(images)

    if layout is None:
        d0 = math.ceil(math.sqrt(images.shape[0]))
        d1 = (d0 - 1) if d0 * (d0 - 1) >= images.shape[0] else d0
    else:
        d0, d1 = layout

    slice_max_s = min(target_size[0] // d0, target_size[1] // d1)
    slice_max_s = (int(slice_max_s), int(slice_max_s))

    if scale_to_fit:
        tf_data = [resize(images[s], slice_max_s, max_bound=True, channel_dim=2) for s in range(len(images))]
    else:
        tf_data = images

    slice_s = list(tf_data[0].shape)
    slice_s = [slice_s[0] + spacing, slice_s[1] + spacing]

    grid_image_size = (int(d0 * slice_s[0]) + 1, int(d1 * slice_s[1]) + 1) + ((3,) if len(images[0].shape) == 3 else ())
    grid_image = torch.ones(grid_image_size)

    for i in range(len(tf_data)):
        iy, ix = int(i // d1), int(i % d1)
        off_y, off_x = iy * slice_s[0], ix * slice_s[1]
        grid_image[off_y: off_y + tf_data[i].shape[0], off_x: off_x + tf_data[i].shape[1]] = tf_data[i]

    return grid_image
