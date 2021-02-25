import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import math

from madpack import log
from madpack import visualizers
from madpack.transforms import resize

from madpack.visualizers.functional import color_label_image, color_overlay


def guess_element(element, target_size):
    """ some heuristics for guessing the data type based on a sample only """
    from madpack import visualizers


    if isinstance(element, (np.int_, int, float)):
        return visualizers.TextData(str(element), target_size)

    element = np.array(element)
    shape = element.shape if hasattr(element, 'shape') else ()

    # with shape
    if len(shape) == 4:

        if len(shape) == 4 and shape[0] == 1 and shape[3] == 3:
            return visualizers.Image(element[0], target_size, channel_dim=2)

        elif len(shape) == 4 and shape[1] in {1,3}:
            return visualizers.Slices(element, target_size, channel_dim=1, maps_dim=0)      
    elif len(shape) == 3:

        if shape[0] == 1:
            return visualizers.ImageGrayscale(element[0], target_size)

        elif shape[0] == 3:  # rgb image
            return visualizers.Image(element, target_size, channel_dim=0)

        elif shape[2] == 3:  # rgb image
            return visualizers.Image(element, target_size, channel_dim=2)

        elif shape[0] > 3 and shape[1] == shape[2]:
            return visualizers.Slices(element, target_size, maps_dim=0)            

        elif shape[0] > 3 and shape[0] == shape[1]:  # sequence
            return visualizers.Slices(element, target_size, maps_dim=2)    

        elif shape[0] > 3:
            return visualizers.Slices(element, target_size, maps_dim=0)
    
    elif len(shape) == 2:

        if np.issubdtype(element.dtype, np.integer):
            return visualizers.LabelImage(element, target_size, maps_dim=None)

        elif np.issubdtype(element.dtype, np.floating):
            return visualizers.Image(element, target_size, color=False)

        elif shape[0] > 10:
            return visualizers.ImageGrayscale(element, target_size)

    elif len(shape) == 1:

        if np.issubdtype(type(element[0]), np.integer):
            return visualizers.TextData(str(element[0]), target_size)

        elif len(shape) == 1:
            return visualizers.DistributionData(element, target_size)

    # if no type matches: FailData
    return visualizers.TextData(f'Failed to guess visualizer\nShape: {shape}', target_size)


def guess_visualizer(samples, target_size, types=None):

    vis_list, info = [], []
    types = types if types is not None else [None for _ in samples]

    for s, type_name in zip(samples, types):

        if type(type_name) == str:
            vis_list += [getattr(visualizers, type_name)(s, target_size)]
        elif callable(type_name):
            vis_list += [type_name(s, target_size)]
        else:
            try:
                vis_list += [guess_element(s, target_size)]
            except Exception as e:
                print(e)
                vis_list += [visualizers.TextData('Failed to guess visualizer', target_size)]

    return vis_list, None


def get_visualizers(samples, dataset, target_size, types=None):

    from madpack import visualizers

    # guess the right visualizer
    if dataset is None or not hasattr(dataset, 'data_types') or dataset.data_types is None:
        return guess_visualizer(samples, target_size, types=types)
    else:
        inp, out = dataset.parse_data_types()

        # overwrite visualizers if explicitly specified
        if dataset.visualizers is not None:
            vis_list = [vis(sample, target_size) for vis, sample in zip(dataset.visualizers, samples)]
            infos = [('a', i, t, s) for i, (t, s) in enumerate(inp + out)]
        else:
            vis_list, infos = [], []
            assert len(samples) == len(inp) + len(out), (f'The number of yielded items ({len(samples)}) must match the '
                                                         f'data_types attribute: {dataset.data_types}')
            for i, (sample, (vis_type, vis_name)) in enumerate(zip(samples, inp + out)):

                if sample is not None:
                    try:
                        if vis_name in dataset.visualization_hooks:
                            sample = dataset.visualization_hooks[vis_name](sample)

                        vis_list += [getattr(visualizers, vis_type)(sample, target_size)]
                        infos += [('input', i, vis_type, vis_name if vis_name is not None else vis_type)]
                    except BaseException as e:
                        shape = sample.shape if hasattr(sample, 'shape') else 'no shape'
                        msg = (f'Failed to initialize visualizer type {vis_type}. Name: {vis_name}, Shape: {shape}\n'
                               'Available visualizers: ' + ', '.join(dir(vis_type)))
                        log.warning(msg)
                        raise e
                else:
                    vis_list += [visualizers.TextData('None', target_size)]
                    infos += [('input', i, vis_type, vis_name if vis_name is not None else vis_type)]

        return vis_list, infos


def plot_data(dataset, end=1, start=0, shuffle=False, height=2.5, elements=None, types=None):

    if type(dataset) in {torch.Tensor, np.ndarray}:
        vi, _ = get_visualizers([dataset], dataset, (300, 300), types=types)
        _plot_visualizer(vi, elements, height)
    else:

        if not shuffle:
            indices = range(start, end)
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            indices = indices[start: end]

        for i in indices:
            sample = dataset[i]
            if len(sample) == 2 and type(sample[0]) in {list, tuple} and type(sample[1]) in {list, tuple}:
                items = [s for s in sample[0]] + [s for s in sample[1]]
                titles = ['in' for _ in sample[0]] + ['out' for _ in sample[1]]
            else:
                items = sample
                titles = [''] * len(sample)
            
            vi, _ = get_visualizers(items, dataset, (300, 300), types=types)
            
            mins, maxs, shapes = ['']*len(items), ['']*len(items), ['']*len(items)
            for j, x in enumerate(items):
            
                try:
                    shapes[j] = tuple(x.shape)
                    mins[j] = round(float(x.min()), 3)
                    maxs[j] = round(float(x.max()), 3)
                except (RuntimeError, AttributeError):
                    pass
                
            dtypes = [x.dtype if hasattr(x, 'dtype') else '' for x in items]
            titles = [f'{t} {s}\n{dt} {mn}-{mx}' for t, s, dt, mn, mx in zip(titles, shapes, dtypes, mins, maxs)]

            _plot_visualizer(vi, elements, height, titles)


def _plot_visualizer(vi, elements, height, titles=None):
    eles = list(enumerate(vi)) if elements is None else [(j, vi[j]) for j in elements]

    w, h = min(16, len(eles) * height), min(height, 16 / len(eles))
    fig, ax = plt.subplots(1, len(eles), figsize=(w, h))

    for k, (j, v) in enumerate(eles):
        axis = ax[k] if len(eles) > 1 else ax
        v.plot(axis)
        # ax[i].text(0, 0, , fontsize=15)
        if titles is not None:
            axis.set_title(titles[j], fontsize=9)


def _prepare_image(img):
    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = img.transpose([1, 2, 0])

    return img


def plot_image(image, axes=None, figsize=(5, 5)):
    fig, ax = plt.subplots(ncols=1, figsize=figsize) if axes is None else (None, axes)
    ax.imshow(_prepare_image(image))


def plot_image_stack(images, figsize=(15, 5), cmap='gray', axes=None, no_stats=False):
    vmin, vmax = min([img.min() for img in images]), max([img.max() for img in images])
    fig, ax = plt.subplots(ncols=len(images), figsize=figsize) if axes is None else (None, axes)

    log.detail(f'min {min([i.min() for i in images])}, max {max([i.max() for i in images])}')

    for i in range(len(images)):

        img = images[i]
        img = _prepare_image(img)

        this_ax = ax[i] if len(images) > 1 else ax
        this_ax.imshow(img, cmap=plt.cm.get_cmap(cmap), vmin=vmin, vmax=vmax)
        this_ax.axis('off')
    
    if fig is not None:
        fig.tight_layout()

    return fig

