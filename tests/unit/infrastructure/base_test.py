from unittest import TestCase
from numpy import testing

from manifold.infrastructure.base import EuclideanDistancesFromDataSet


class EuclideanDistancesFromDataSetTest(TestCase):
    def test_basic(self):
        data_set = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]

        expected = {
            0: {1: 1, 2: 1.4142135623730951, 3: 1.7320508075688772},
            1: {2: 1, 3: 1.4142135623730951},
            2: {3: 1},
        }

        actual = EuclideanDistancesFromDataSet(data_set).run()

        self.assertDictEqual(expected, actual)
