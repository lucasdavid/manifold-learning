from unittest import TestCase
from sklearn import datasets, manifold

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from manifold.infrastructure import Displayer


class DisplayerTest(TestCase):
    def test_random_generation(self):
        """Tests if the Displayer class is presenting 4 graphics correctly (manual checking).
        """
        points = 1000
        data, color = datasets.make_swiss_roll(points, random_state=0)
        neighbors = 10

        d = Displayer(points=points, neighbors=neighbors)

        d \
            .load(data, color, title='Graphic I') \
            .load(data, color, title='Graphic II') \
            .load(data, color, title='Graphic III') \
            .load(data, color, title='Graphic IV') \
            .show()

    def test_similar_graphics(self):
        """Tests if Displayer class is presenting a similar graphic from the one printed
        by the hard-coded lines bellow (manual checking).
        """
        points = 1000
        data, color = datasets.make_swiss_roll(points, random_state=0)
        neighbors = 10
        to_dimension = 2

        result = manifold.Isomap(neighbors, to_dimension).fit_transform(data)

        # Expected printing...
        Axes3D
        fig = plt.figure(figsize=(15, 8))
        plt.suptitle("Expected image", fontsize=14)
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
        ax = fig.add_subplot(122)
        plt.scatter(result[:, 0], result[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.title("SKLearn's Isomap")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

        # Actual printing...
        Displayer(title="Actual image", points=points, neighbors=neighbors) \
            .load(data, color, title='Graphic I') \
            .load(result, color, title='SKLearn\'s Isomap') \
            .show()
