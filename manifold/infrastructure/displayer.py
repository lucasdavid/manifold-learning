import copy
import math
import numpy as np
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

    def load(self, title, data, color=None):
        self.items.append(DisplayItem(title, data, color))

        return self

    def render(self):
        fig = plt.figure(figsize=(16, 9))
        plt.suptitle(self.parameters)

        count = len(self.items)
        items_in_row = math.ceil(math.sqrt(count))
        rows_count = math.ceil(count / items_in_row)

        for i, item in enumerate(self.items):
            samples, dimension = item.data.shape

            # Consider, at most, the 3 first components.
            components = [item.data[:, i] for i in range(min(dimension, 3))]

            if dimension == 1:
                components.append(np.zeros((samples, 1)))

            kwargs = {}
            if dimension > 2:
                kwargs['projection'] = '3d'

            ax = fig.add_subplot(
                rows_count * 100 +
                items_in_row * 10 +
                1 + i, **kwargs)

            kwargs = {}
            if item.color is not None:
                kwargs['c'] = item.color

            ax.scatter(*components, **kwargs)
            if item.title:
                plt.title(item.title)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')

            if dimension > 2:
                ax.zaxis.set_major_formatter(NullFormatter())
                ax.view_init(4, -72)

        plt.show()

        return self
