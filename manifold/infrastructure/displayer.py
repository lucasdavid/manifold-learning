import copy
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter


class DisplayItem(object):
    def __init__(self, title, data, color):
        self.title = title
        self.data = copy.deepcopy(data)
        self.color = copy.deepcopy(color)


class Displayer(object):
    def __init__(self, **kwargs):
        self.items = []
        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])

    def load(self, title, data, color):
        self.items.append(DisplayItem(title, data, color))

        return self

    def render(self):
        fig = plt.figure(figsize=(16, 9))
        plt.suptitle(self.parameters)

        count = len(self.items)
        items_in_row = count // 2
        rows_count = math.ceil(count / items_in_row)

        for i, item in enumerate(self.items):
            is_3d = item.data.shape[1] > 2

            args = [item.data[:, 0], item.data[:, 1]]
            if is_3d:
                args.append(item.data[:, 2])

            kwargs = {}
            if is_3d:
                kwargs['projection'] = '3d'
                Axes3D

            ax = fig.add_subplot(
                rows_count * 100 +
                items_in_row * 10 +
                1 + i, **kwargs)

            ax.scatter(*args, c=item.color, cmap=plt.cm.Spectral)
            if item.title:
                plt.title(item.title)

            if is_3d:
                ax.view_init(4, -72)
            else:
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(NullFormatter())
                plt.axis('tight')

        plt.show()

        return self
