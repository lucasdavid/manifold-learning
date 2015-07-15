from unittest import TestCase
from sklearn import datasets

from manifold.services import Displayer


class DisplayerTest(TestCase):
    def test_random_generation(self):
        points = 1000
        X, color = datasets.make_swiss_roll(points, random_state=0)
        neighbors = 10

        d = Displayer(points=points, neighbors=neighbors)

        d.load(X, color).render()
