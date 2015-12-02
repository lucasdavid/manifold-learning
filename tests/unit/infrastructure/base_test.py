from unittest import TestCase

import numpy as np

from sklearn import datasets, manifold
from manifold.infrastructure.base import EuclideanDistancesFromDataSet, class_stress


class EuclideanDistancesFromDataSetTest(TestCase):
    def test_basic(self):
        data_set = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ])

        expected = {
            0: {1: 1, 2: 1.4142135623730951, 3: 1.7320508075688772},
            1: {2: 1, 3: 1.4142135623730951},
            2: {3: 1},
        }

        actual = EuclideanDistancesFromDataSet(data_set).run()

        self.assertDictEqual(expected, actual)


class ClassStressTest(TestCase):
    def test_basic(self):
        X, target = datasets.make_swiss_roll(n_samples=1000, random_state=0)
        Y = manifold.Isomap().fit_transform(X)
        result = class_stress(X, Y, target, n_jobs=8)

        self.fail(result)
