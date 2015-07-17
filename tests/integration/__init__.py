import numpy as np

from unittest import TestCase

from manifold.learning import algorithms
from manifold.infrastructure import Displayer


class IsomapPrintingTest(TestCase):
    def setUp(self):
        self.test_data = [
            [0, 93, 82, 133],
            [93, 0, 52, 60],
            [82, 52, 0, 111],
            [133, 60, 111, 0],
        ]

    def test_rendering(self):
        i = algorithms.Isomap(self.test_data, to_dimension=2)

        result = i.execute()

        Displayer() \
            .load(result, np.random.rand(4)) \
            .render()
