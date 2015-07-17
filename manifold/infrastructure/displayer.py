import copy
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class DisplayItem(object):
    def __init__(self, data, color, is_3d):
        self.data = copy.deepcopy(data)
        self.color = copy.deepcopy(color)
        self.is_3d = is_3d


class Displayer(object):
    def __init__(self, **kwargs):
        self.items = []
        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])

    def load(self, data, color, is_3d=False):
        self.items.append(DisplayItem(data, color, is_3d))

        return self

    def render(self):
        fig = plt.figure(figsize=(16, 9))
        plt.suptitle("Manifold Learning with " + self.parameters, fontsize=14)

        count = len(self.items)
        items_in_row = count // 2 + 1
        rows_count = math.ceil(count / items_in_row)

        for i, item in enumerate(self.items):
            is_3d = len(item.data.shape) == 3

            args = [item.data[:, 0], item.data[:, 1]]
            if is_3d:
                args.append(item.data[:, 2])

            kwargs = {}
            if is_3d:
                kwargs['projection'] = '3d'

            ax = fig.add_subplot(
                rows_count * 100 +
                items_in_row * 10 +
                1 + i, **kwargs)

            ax.scatter(*args, c=item.color, cmap=plt.cm.Spectral)
            if is_3d:
                ax.view_init(4, -72)

        plt.show()

        return self
