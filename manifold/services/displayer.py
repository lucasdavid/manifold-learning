import copy
import math
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class DisplayItem(object):
    def __init__(self, data, color):
        self.data = copy.deepcopy(data)
        self.color = copy.deepcopy(color)


class Displayer(object):
    def __init__(self, **kwargs):
        self.items = []
        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])

    def load(self, data, color):
        self.items.append(DisplayItem(data, color))

        return self

    def render(self):
        Axes3D

        fig = plt.figure(figsize=(16, 9))
        plt.suptitle("Manifold Learning with " + self.parameters, fontsize=14)

        count = len(self.items)
        items_in_row = count // 2 + 1
        rows_count = math.ceil(count / items_in_row)

        for i, item in enumerate(self.items):
            ax = fig.add_subplot(
                rows_count * 100 +
                items_in_row * 10 +
                1 + i, projection='3d')

            ax.scatter(item.data[:, 0], item.data[:, 1], item.data[:, 2], c=item.color, cmap=plt.cm.Spectral)
            ax.view_init(4, -72)

        plt.show()

        return self
