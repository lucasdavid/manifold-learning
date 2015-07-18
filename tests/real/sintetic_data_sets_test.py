import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from unittest import TestCase
from sklearn import datasets

from manifold.learning import algorithms
from manifold.infrastructure import Displayer


class ISOMAPTest(TestCase):
    def test_swiss_roll(self):
        data, color = datasets.make_swiss_roll(n_samples=200, random_state=0)

        result = algorithms \
            .Isomap(data, color=color, k=10, to_dimension=2) \
            .run()

        fig = plt.figure(figsize=(15, 8))
        plt.suptitle("Manifold Learning with %i points, %i neighbors"
                     % (1000, 100), fontsize=14)

        Axes3D
        ax = fig.add_subplot(251, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
        ax.view_init(4, -72)
        plt.show()

        Displayer() \
            .load(data, color, is_3d=True) \
            .load(result, color) \
            .render()
