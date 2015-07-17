import numpy as np
from unittest import TestCase
from numpy import testing

from manifold.learning import algorithms


class FloydWarshallTest(TestCase):
    def test_digraph(self):
        # https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
        distance = [
            [0, 0, -2, 0],
            [4, 0, 3, 0],
            [0, 0, 0, 2],
            [0, -1, 0, 0],
        ]

        expected = [
            [0, -1, -2, 0],
            [4, 0, 2, 4],
            [5, 1, 0, 2],
            [3, -1, 1, 0],
        ]

        f = algorithms.FloydWarshall(distance_matrix=distance)
        actual = f.execute()

        testing.assert_array_almost_equal(actual, expected)


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

        i = algorithms.Isomap(proximity_matrix, to_dimension=2)

        actual = i.execute()

        testing.assert_array_almost_equal(actual, expected, decimal=3)
