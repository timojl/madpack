import numpy as np


def color_label_image(labels, n_colors=None):
    """ Create a color image from a label image consisting of category ids per pixel. """
    n_colors = n_colors if n_colors is not None else max(20, labels.max() + 1)

    from skimage.color import lab2rgb

    colors = np.uint8(lab2rgb(np.dstack([
        35 + np.sin(np.arange(0, 1, 1 / n_colors) * 2 * np.pi) * 20,
        np.sin(np.arange(-1, 1, 2 / n_colors) * 1.5 * np.pi) * 50,
        np.sin(np.arange(-1, 1, 2 / n_colors) * 3 * np.pi) * 50
    ])) * 255)[0]

    # gray for index 0
    colors = np.concatenate((np.array([[190, 190, 190]], 'uint8'), colors), 0)
    return colors[labels.ravel()].reshape(labels.shape + (3,))    


def color_overlay(img, maps, h_shift, indices, one_hot, n_colors=None, start_color=0, intensities=None, scale_to_image=False):
    n_colors = len(indices) if n_colors is None else n_colors

    import torch
    from skimage.color import hsv2rgb, rgb2gray, gray2rgb
    from madpack.transforms import resize

    color_palette = hsv2rgb(np.dstack([
        np.arange(h_shift, 1, 1 / n_colors),
        0.7 * np.ones(n_colors),
        0.3 * np.ones(n_colors)
    ]))[0]

    out = gray2rgb(rgb2gray(img))  # desaturate
    out *= 0.5

    for i, idx in enumerate(indices):
        m = maps == idx if one_hot else maps[:, :, idx]

        if scale_to_image:
            m = resize(torch.tensor(m), img.shape[:2]).numpy()

        # m = np.clip(m, 0, 1)
        col = color_palette[start_color + i]

        if intensities is not None:
            col = col * intensities[i]

        out = np.clip(out + col * m[:, :, None], 0, 1)

    out = (255 * out).astype('uint8')
    return out
