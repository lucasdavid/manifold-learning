import numpy as np
from unittest import TestCase
from numpy import testing

from sklearn import datasets
from manifold.learning import algorithms


class KNearestNeighborsTest(TestCase):
    def test_basic(self):
        m = [
            [ 0, 10, 2,  3],
            [10,  0, 2, 15],
            [ 2,  2, 0,  3],
            [ 3, 15, 3,  0],
        ]

        k = 2

        expected = [
            [ 0, 0, 2, 3],
            [10, 0, 2, 0],
            [ 2, 2, 0, 0],
            [ 3, 0, 3, 0],
        ]

        actual = algorithms \
            .KNearestNeighbors(distance_matrix=m, k=k) \
            .run()

        self.assertIsNotNone(actual)
        testing.assert_array_almost_equal(actual, expected)


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
        actual = f.run()

        testing.assert_array_almost_equal(actual, expected)


class MDSTest(TestCase):
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

        i = algorithms.MDS(proximity_matrix, to_dimension=2)

        actual = i.run()

        testing.assert_array_almost_equal(actual, expected, decimal=3)
