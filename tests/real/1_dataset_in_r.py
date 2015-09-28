import os

from unittest import TestCase
from manifold.infrastructure import Retriever, Displayer
from manifold.learning import algorithms


class GlassDataSetTest(TestCase):
    def test_display_dimensions(self):
        data_dir = 'datasets/'
        data_set = 'glass/glass.data'
        file = os.path.join(data_dir, data_set)

        print('Displaying data set {%s} in the Rn' % file)

        glass = Retriever(file, delimiter=',')

        # Glass has the samples' ids in the first column.
        glass.split_column(0)
        # Additionally, its last column represents the target feature.
        glass.split_target()

        data, c = glass.retrieve()
        reduced_data = algorithms.Isomap(data, e=20).run()

        d = Displayer(title=data_set)

        # Scatter all dimensions (3-by-3), using as many graphs as necessary.
        for begin in range(0, glass.features_count, 3):
            end = min(glass.features_count, begin + 3)
            d.load(data[:, begin:end], color=c, title='Dimensions: d e [%i, %i]' % (begin + 1, end))

        d \
            .load('Reduced glass data-set', reduced_data, c) \
            .render()
