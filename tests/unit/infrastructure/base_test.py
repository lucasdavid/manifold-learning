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

        _14 = 1.414214
        _17 = 1.732051

        expected = [
            [  0,   1, _14, _17],
            [  1,   0,   1, _14],
            [_14,   1,   0,   1],
            [_17, _14,   1,   0],
        ]

        actual = EuclideanDistancesFromDataSet(data_set).run()

        testing.assert_array_almost_equal(actual, expected)
