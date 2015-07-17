import numpy as np
from unittest import TestCase
from numpy import testing

from manifold.learning.algorithms import Isomap


class IsomapTest(TestCase):
    def test_wickelmaier(self):
        proximity_matrix = [
            [0, 93, 82, 133],
            [93, 0, 52, 60],
            [82, 52, 0, 111],
            [133, 60, 111, 0],
        ]

        expected = [
            [-62.831, 32.97448],
            [18.403, -12.02697],
            [-24.960, -39.71091],
            [69.388, 18.76340],
        ]

        i = Isomap(proximity_matrix, to_dimension=2)

        actual = i.execute()

        testing.assert_array_almost_equal(actual, expected, decimal=3)
