import torch
from torch.nn import functional as nnf


def soft_histogram(a, size=20, sigma=0.02):
    """ Computes a histogram in a differentiable way.
        a: tensor of batch_size x ... x ...
    """
    # const_fac = 1 / sigma * math.sqrt(2 * 3.141)
    bs = a.size(0)

    a_flat = a.view(bs, -1)
    img = torch.empty(size, bs, a_flat.size(1))

    # interestingly, this works better for linspace [0,1] instead of [1/size, 1-1/size]
    for i, x in enumerate(torch.linspace(0, 1, size)):
        pow2 = torch.pow((x - a_flat) / sigma, 2)
        img[i] = torch.exp(-0.5 * pow2)

    img = img.sum(2)
    img = img / img.sum(0)[None, :]
    img = torch.transpose(img, 0, 1)

    return img
