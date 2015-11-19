from time import time
from unittest import TestCase
from sklearn import datasets, manifold

from manifold.infrastructure import Displayer
from manifold.learning.algorithms import Isomap


class SKLearnIsomapTest(TestCase):
    def test_swiss_roll(self):
        samples = 1000
        neighbors = 10
        n_components = 2

        data, c = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        displayer = Displayer(title="Isomap algorithms comparison") \
            .load(title="Swiss roll from %i samples." % (samples,), data=data, color=c)

        start = time()
        result = manifold.Isomap(neighbors, n_components).fit_transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="SKLearn's Isomap with %i neighbors, taking %.1fs." % (neighbors, elapsed),
                data=result,
                color=c)

        start = time()
        result = Isomap(k=neighbors, n_components=n_components).transform(data)
        elapsed = time() - start

        displayer \
            .load(
                title="Isomap with %i neighbors, taking %.1fs" % (neighbors, elapsed),
                data=result,
                color=c)
        displayer.show()
