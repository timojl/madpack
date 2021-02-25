import torch


def build_theta_2d(scale, mx, my):
    """ build theta matrix for 2d spatial transformer """
    theta = torch.zeros(mx.size(0), 2, 3, device=mx.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = 1 - 2 * mx
    theta[:, 1, 2] = 1 - 2 * my
    return theta


def build_theta_3d(scale_spatial, scale_time, mx, my, mz, pos=False):
    """ build theta matrix for 3d spatial transformer """
    theta = torch.zeros(mx.size(0), 3, 4, device=mx.device)
    theta[:, 0, 0] = scale_spatial
    theta[:, 1, 1] = scale_spatial
    theta[:, 2, 2] = scale_time
    theta[:, 0, 3] = 1 - 2 * mx if not pos else 2 * mx - 1
    theta[:, 1, 3] = 1 - 2 * my if not pos else 2 * my - 1
    theta[:, 2, 3] = 1 - 2 * mz if not pos else 2 * mz - 1
    return theta


def first_moment_to_scalar(tensor, dims, norm_out=True):

    projs = []
    for d in dims:
        proj = tensor.sum([i for i in dims if i != d])

        proj = torch.softmax(proj, -1)
        proj = (proj * torch.linspace(0, 1 if norm_out else tensor.size(d) - 1, tensor.size(d), device=tensor.device))
        proj = proj.sum(-1)

        projs += [proj]

    return torch.stack(projs, dim=-1)
