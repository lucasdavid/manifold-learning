from unittest import TestCase
from sklearn import datasets, manifold
from time import time

from manifold.learning import algorithms
from manifold.infrastructure import Displayer


class IsomapTest(TestCase):
    def test_swiss_roll(self):
        samples = 1000
        neighbors = 10
        epsilon = 5
        to_dimension = 2

        data, c = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        displayer = Displayer(title="Isomap algorithms comparison") \
            .load(title="Swiss roll from %i samples." % (samples,), data=data, color=c)

        start = time()
        result = manifold.Isomap(neighbors, to_dimension, path_method='D').fit_transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="SKLearn's Isomap (%i neighbors, dijkstra, taking %.1fs)" % (neighbors, elapsed),
                data=result,
                color=c)

        start = time()
        result = manifold.Isomap(neighbors, to_dimension, path_method='FW').fit_transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="SKLearn's Isomap (%i neighbors, floyd-warshall, taking %.1fs)" % (neighbors, elapsed),
                data=result,
                color=c)

        start = time()
        result = algorithms \
            .Isomap(data, nearest_method='e', e=epsilon, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="E-Isomap (epsilon=%i, dijkstra, %.1fs)" % (epsilon, elapsed),
            data=result,
            color=c)

        start = time()
        result = algorithms \
            .Isomap(data, nearest_method='e', shortest_path_method='fw', e=epsilon, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="E-Isomap (epsilon=%i, floyd-warshall, %.1fs)" % (epsilon, elapsed),
            data=result,
            color=c)

        start = time()
        result = algorithms \
            .Isomap(data, k=neighbors, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="K-Isomap (%i neighbors, dijkstra, %.1fs)" % (neighbors, elapsed),
            data=result,
            color=c)

        start = time()
        result = algorithms \
            .Isomap(data, k=neighbors, to_dimension=to_dimension) \
            .run()
        elapsed = time() - start

        displayer.load(
            title="K-Isomap (%i neighbors, floyd-warshall, %.1fs)" % (neighbors, elapsed),
            data=result,
            color=c)

        displayer.render()
