from unittest import TestCase

import numpy as np
from numpy import testing
from nose_parameterized import parameterized

from manifold.infrastructure import errors, Retriever


class RetrieverTest(TestCase):
    def test_retrieve_brainwave(self):
        retriever = Retriever('datasets/brainwave/plrx.txt', target_column=-1)

        data, target = retriever.load().split_target().retrieve()

        # Assert number of samples.
        self.assertEqual(182, len(data))

        # Assert number of features.
        self.assertEqual(12, len(data[1]))

        # Assert number of samples in target array.
        self.assertEqual(182, len(target))

    def test_retrieve_nonexistent_file(self):
        fake_file = 'nonexistentfile.data'

        r = Retriever(fake_file)

        with self.assertRaises(errors.RetrieverError):
            r.load()

    @parameterized.expand([
        ([[1, 2, 3], [4, 5, 6]], -1, [[1, 2], [4, 5]], [3, 6]),
        ([[1, 2, 3], [4, 5, 6]], 0, [[2, 3], [5, 6]], [1, 4])
    ])
    def test_split_target(self, data, target_column, expected_data, expected_target):
        r = Retriever('nonexistentfile.data')
        r._data = np.array(data)
        actual_data, actual_target = r.split_target(target_column).retrieve()

        testing.assert_array_equal(actual_data, np.array(expected_data))
        testing.assert_array_equal(expected_target, np.array(expected_target))
