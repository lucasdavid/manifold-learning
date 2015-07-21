import numpy as np

from unittest import TestCase
from sklearn import datasets, manifold
from time import time

from manifold.learning import algorithms
from manifold.infrastructure import Displayer


class SmallDataSetsTest(TestCase):
    def test_canonical_data(self):
        data = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ])

        c = np.array([0, 0, 1, 1])

        neighbors = 2
        epsilon = 1.5
        to_dimension = 2

        displayer = Displayer(title="Isomap algorithms comparison") \
            .load(title="Canonical data set.", data=data, color=c)

        start = time()
        result = manifold.Isomap(neighbors, to_dimension).fit_transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="SKLearn's Isomap with %i neighbors, taking %.1fs." % (neighbors, elapsed),
                data=result,
                color=c)

        start = time()
        result = algorithms \
            .Isomap(data, color=c, nearest_method='e', e=epsilon, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="My E-Isomap with epsilon %i, taking %.1fs." % (epsilon, elapsed),
            data=result,
            color=c)

        start = time()
        result = algorithms \
            .Isomap(data, color=c, k=neighbors, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="My K-Isomap with %i neighbors, taking %.1fs." % (neighbors, elapsed),
            data=result,
            color=c)

        displayer.render()

    def test_random_matrix(self):
        data = np.trunc(5 + 10 * np.random.rand(10, 3))

        neighbors = 2
        epsilon = 5.
        to_dimension = 2

        displayer = Displayer(title="Isomap algorithms comparison") \
            .load(title="Canonical data set.", data=data)

        start = time()
        result = manifold.Isomap(neighbors, to_dimension).fit_transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="SKLearn's Isomap with %i neighbors, taking %.1fs." % (neighbors, elapsed),
                data=result)

        start = time()
        result = algorithms \
            .Isomap(data, nearest_method='e', e=epsilon, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="My E-Isomap with epsilon %i, taking %.1fs." % (epsilon, elapsed),
            data=result)

        start = time()
        result = algorithms \
            .Isomap(data, k=neighbors, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="My K-Isomap with %i neighbors, taking %.1fs." % (neighbors, elapsed),
            data=result)

        displayer.render()
