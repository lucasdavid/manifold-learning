import numpy as np

from unittest import TestCase

from manifold.services import errors, Retriever


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

    def test_split_target(self):
        fake_file = 'nonexistentfile.data'

        r = Retriever(fake_file)

        r._data = np.array([[1, 2, 3], [4, 5, 6]])
        data, target = r.split_target().retrieve()

        self.assertTrue((data == np.array([[1, 2], [4, 5]])).all())
        self.assertTrue((target == np.array([3, 6])).all())

        r._data = np.array([[1], [4]])
        data, target = r.split_target().retrieve()

        self.assertTrue((data == np.array([])).all())
        self.assertTrue((target == np.array([1, 4])).all())

        r._data = np.array([[1, 2, 3], [4, 5, 6]])
        data, target = r.split_target(0).retrieve()

        self.assertTrue((data == np.array([[2, 3], [5, 6]])).all())
        self.assertTrue((target == np.array([1, 4])).all())
