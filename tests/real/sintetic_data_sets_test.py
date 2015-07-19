from unittest import TestCase
from sklearn import datasets, manifold
from time import time

from manifold.learning import algorithms
from manifold.infrastructure import Displayer


class ISOMAPTest(TestCase):
    def test_swiss_roll(self):
        samples = 1000
        neighbors = 10
        epsilon = 5
        to_dimension = 2

        data, c = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        displayer = Displayer(title="Isomap algorithms comparison") \
            .load(title="Swiss roll from %i samples." % (samples,), data=data, color=c)

        t0 = time()
        my_result = algorithms \
            .Isomap(data, color=c, nearest_method='e', e=epsilon, to_dimension=to_dimension) \
            .run()
        t1 = time()

        displayer.load(title="My Isomap with %i neighbors." % (neighbors,), data=my_result, color=c)

        t0 = time()
        sklearn_result = manifold.Isomap(neighbors, to_dimension).fit_transform(data)
        t1 = time()

        displayer \
            .load(title="SKLearn's Isomap with %i neighbors." % (neighbors,), data=sklearn_result, color=c) \
            .render()
