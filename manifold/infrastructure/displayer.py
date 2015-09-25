import math
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter


class Displayer(object):
    def __init__(self, **kwargs):
        self.items = []
        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])

    def load(self, data, color=None, title=None):
        # Always copy the data, and, of course, only the first three dimensions.
        self.items.append((data[:, :3], color, title))

        return self

    def render(self):
        # Assert that there is at least one graph to show.
        assert self.items

        fig = plt.figure(figsize=(16, 9))
        plt.suptitle(self.parameters)

        count = len(self.items)
        items_in_row = math.ceil(math.sqrt(count))
        rows_count = math.ceil(count / items_in_row)

        for i, item in enumerate(self.items):
            data, color, title = item
            samples, dimension = data.shape

            # Grab data set components. It necessarily has 3 dimensions, as it was cut during load().
            components = [data[:, i] for i in range(dimension)]

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
            if color is not None:
                kwargs['c'] = color

            ax.scatter(*components, cmap=plt.cm.jet, **kwargs)
            if title:
                plt.title(title)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')

            if dimension > 2:
                ax.zaxis.set_major_formatter(NullFormatter())
                ax.view_init(4, -72)

        plt.show()

        return self
