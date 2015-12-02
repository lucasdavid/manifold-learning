import datetime
import math
import os
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Displayer(object):
    colors = [plt.cm.jet, plt.cm.rainbow, plt.cm.hsv, plt.cm.CMRmap]

    def __init__(self, **kwargs):
        self.aspect = kwargs.pop('aspect', (20, -40))
        self.rows = kwargs.pop('rows', None)
        self.columns = kwargs.pop('columns', None)
        self.parent_is_plotting = kwargs.pop('plotting', True)
        self.saving_folder = kwargs.pop('folder', '/home/ldavid/Desktop/reports')

        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])
        self.items = []

    def load(self, data, color=None, title=None):
        if self.parent_is_plotting:
            # Always copy the data, and, of course, only the first three dimensions.
            # Doesnt do anything if parent isn't plotting, though, as it would wasting memory.
            self.items.append((data[:, :3], color, title))

        return self

    def show(self):
        """Show graphs.

        Returns
        -------
        self.
        """
        if self.parent_is_plotting:
            # Ignores calls if parent is not plotting.
            self._process_figure()
            plt.show()

        return self

    def save(self, name=None):
        if self.parent_is_plotting:
            # Ignore calls if parent is not plotting.
            figure = self._process_figure()

            name = (name or str(datetime.datetime.now())).replace(':', '.')
            name = os.path.join(self.saving_folder, name)
            name += '.png'

            plt.savefig(name, bbox_inches='tight')
            plt.close(figure)

        return self

    def _process_figure(self):
        # Assert that there is at least one graph to show.
        assert self.items, 'no graphs to render.'

        figure = plt.figure(figsize=(16, 9))
        plt.suptitle(self.parameters)

        count = len(self.items)
        columns = self.columns or math.ceil(math.sqrt(count))
        rows = self.rows or math.ceil(count / columns)

        color_map = random.choice(self.colors)

        for i, item in enumerate(self.items):
            data, color, title = item
            samples, dimension = data.shape

            # Grab data set components. It necessarily has 3 dimensions, as it was cut during load().
            components = [data[:, i] for i in range(dimension)]

            if dimension == 1:
                components.append(np.zeros((samples, 1)))

            if color is None:
                color = np.zeros(samples)

            kwargs = {'projection': '3d', 'alpha': .2} if dimension > 2 else {}
            ax = figure.add_subplot(rows, columns, 1 + i, **kwargs)

            ax.scatter(*components, **{
                'c': color,
                's': 50,
                'cmap': color_map
            })

            if title:
                plt.title(title)

            plt.axis('tight')

            if dimension > 2:
                ax.view_init(*self.aspect)

        return figure

    def dispose(self):
        self.items = []

        return self
