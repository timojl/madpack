import numpy as np


def render_polygons(annotations, img_size, object_id_to_index, first_depth_index_is_bg=False, normalize=False, target_size=None, target_dtype='bool'):
    from skimage.draw import polygon

    """
    Computes a tensor from polygons. Expects input to be in the this form:
    `((img_height, img_width), [(object_id, [[x1, y1, x2, y2, ...], [optional further polygon]], (object_id, [[...]])])`
    `object_id_to_index` needs to map every occurring object_id to an index and there should be gaps in object_id_to_index.
    """

    if target_size is None:
        target_size = img_size
        scale = 1
    else:
        scale = target_size[0] / img_size[0], target_size[1] / img_size[1]
        raise NotImplementedError

    tensor = np.zeros((target_size[0], target_size[1], len(object_id_to_index)), target_dtype)

    for object_class_id, polygons in annotations:
        for p in polygons:
            x = np.array(p[0::2])
            y = np.array(p[1::2])
            rr, cc = polygon(y, x, shape=target_size)
            tensor[rr, cc, object_id_to_index[object_class_id]] = True

    if first_depth_index_is_bg:  # if a pixel has no class it is background
        tensor[:, :, 0] = 1 - tensor[:, :, 1:].sum(2)

    if normalize:
        tensor = tensor / tensor.sum(2)[:, :, np.newaxis]

    return tensor