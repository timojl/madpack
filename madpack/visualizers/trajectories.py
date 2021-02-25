from madpack.visualizers.base import VisualizerBase
import numpy as np


def plot_trajectories(*trajectory_sets, fac=2, image_size=(64, 64)):
    from skimage.draw import line_aa

    base_colors = np.array([
        [0.1, 0.4, 0.8],
        [0.8, 0.3, 0.3],
        [0.4, 0.95, 0.2],
    ])

    out = []
    for trajectories in trajectory_sets:

        img = np.zeros(image_size + (3,), dtype=np.float)

        for k in range(len(trajectories)):

            trajectory = np.array(trajectories[k])

            y, x = trajectory[0, 0] * fac, trajectory[0, 1] * fac
            img[y - 2:y + 1, x - 2:x + 1] = base_colors[k]
            for i in range(1, len(trajectory)):
                a = trajectory[i - 1] * fac
                b = trajectory[i] * fac

                rr, cc, val = line_aa(a[0], a[1], b[0], b[1])
                rr, cc, val = rr[1:], cc[1:], val[1:]

                img[rr, cc, 0] += base_colors[k, 0] * val
                img[rr, cc, 1] += base_colors[k, 1] * val
                img[rr, cc, 2] += base_colors[k, 2] * val

        out += [np.clip(img, 0, 1)]

    if len(out) == 1:
        out = out[0]

    return out


class Trajectories(VisualizerBase):
    """
    `maps_dim` indicates the dimension along which the maps are stored.
    `maps_dim` and `channel_dim` ignore the batch dimension
    """
    def __init__(self, data_item, target_size, frame_size=None):
        super().__init__(data_item, target_size)
        self.frame_size = frame_size

    def plot(self, ax):
        img = self.as_image()
        ax.imshow(img, vmin=0, vmax=1)

    def as_image(self):

        trajectories = np.expand_dims(self.data_item,0) if self.data_item.ndim == 2 else self.data_item

        max_val = self.data_item.max()
        frame_size = (max_val + 10, max_val + 10) if self.frame_size is None else self.frame_size
        traj_img = plot_trajectories(trajectories, image_size=frame_size, fac=1)
        return traj_img

    def get_visdom_data(self):
        return self.as_image(), 'image'
