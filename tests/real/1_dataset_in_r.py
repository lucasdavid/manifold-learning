import os

from unittest import TestCase
from manifold.infrastructure import Retriever, Displayer


class GlassDataSetTest(TestCase):
    def test_display_dimensions(self):
        data_sets_dir = '../datasets'
        data_set = 'glass/glass.data'

        file = os.path.join(data_sets_dir, data_set)

        print('Displaying data set {%s} in the Rn' % file)

        glass = Retriever(file, delimiter=',')

        # Glass has the samples' ids in the first column.
        glass.split_column(0)
        # Additionally, its last column represents the target feature.
        glass.split_target()

        data, color = glass.retrieve()

        d = Displayer(title=data_set)

        # Scatter all dimensions (3-by-3), using as many graphs as necessary.
        for begin in range(0, glass.features, 3):
            end = min(glass.features, begin + 3)
            d.load('Dimensions: d e [%i, %i]' % (begin + 1, end), data[:, begin:end], color=color)

        d.render()
