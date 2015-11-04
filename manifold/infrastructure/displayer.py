import math
import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix


class Displayer(object):
    def __init__(self, **kwargs):
        self.aspect = kwargs.pop('aspect', (20, -40))
        self.rows = kwargs.pop('rows', None)
        self.columns = kwargs.pop('columns', None)

        self.parameters = ', '.join(['%s: %s' % (k, str(v)) for k, v in kwargs.items()])
        self.items = []

    def load(self, data, color=None, title=None):
        # Always copy the data, and, of course, only the first three dimensions.
        self.items.append((data[:, :3], color, title))

        return self

    def render(self):
        self._process_figure()
        plt.show()

        return self

    def save(self, name=None):
        figure = self._process_figure()

        name = name or '%s.png' % str(datetime.datetime.now()).replace(':', '.')
        plt.savefig(name, bbox_inches='tight')
        plt.close(figure)

        return self

    def _process_figure(self):
        # Assert that there is at least one graph to show.
        assert self.items, 'nothing graphs to render.'

        figure = plt.figure(figsize=(16, 9))
        plt.suptitle(self.parameters)

        count = len(self.items)
        columns = self.columns or math.ceil(math.sqrt(count))
        rows = self.rows or math.ceil(count / columns)

        for i, item in enumerate(self.items):
            data, color, title = item
            samples, dimension = data.shape

            # Grab data set components. It necessarily has 3 dimensions, as it was cut during load().
            components = [data[:, i] for i in range(dimension)]

            if dimension == 1:
                components.append(np.zeros((samples, 1)))

            if color is None:
                color = np.zeros(samples)

            kwargs = {'projection': '3d'} if dimension > 2 else {}

            ax = figure.add_subplot(
                rows * 100 +
                columns * 10 +
                1 + i, **kwargs)

            ax.scatter(*components, **{
                'c': color,
                's': 50,
                'cmap': plt.cm.rainbow
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

    @classmethod
    def confusion_matrix_for(cls, target_test, target_predicted, title='Confusion matrix'):
        cm = confusion_matrix(target_test, target_predicted)
        np.set_printoptions(precision=2)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.show()
