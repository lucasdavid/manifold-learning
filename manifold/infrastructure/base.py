import abc
import copy
import time

import numpy as np
from scipy.spatial.distance import pdist


class Task(object, metaclass=abc.ABCMeta):
    def __init__(self, copying=False, **kwargs):
        self.data = {}
        self.store(copying, **kwargs)

    def store(self, copying=False, **kwargs):
        self.data.update(copying and copy.deepcopy(kwargs) or kwargs)
        return self

    def dispose(self):
        self.data = {}
        return self

    def run(self):
        raise NotImplementedError

    @property
    def verbose(self):
        return 'verbose' in self.data and self.data['verbose']


class EuclideanDistancesFromDataSet(Task):
    def __init__(self, data_set, verbose=False):
        """Creates a upper triangular matrix of distances between each sample of a given data set.

        Parameters
        ----------
        data_set
            The data set which contains the points used in the distance finding.

        verbose
            Flag indicating if info should be outputted.
        """
        super().__init__(data_set=data_set, verbose=verbose, copying=False)

    def run(self):
        start = time.time()

        data_set = self.data['data_set']
        samples = data_set.shape[0]

        # Find the distance between each pair of points.
        d = pdist(data_set)

        distances = {}

        for i in range(samples - 1):
            distances[i] = {i + n + 1: d for n, d in enumerate(d[0:samples - i - 1])}
            d = d[samples - i - 1:]

        if self.verbose:
            print('Task EuclideanDistancesFromDataSet took %.2f s.' % (time.time() - start))
        return distances


class Reducer(Task, metaclass=abc.ABCMeta):
    _stress = None
    embedding = None

    @property
    def stress(self):
        return self._stress

    def transform(self, data):
        self.store(data=data)
        self.run()

        return self.embedding

    def dispose(self):
        super().dispose()
        del self.embedding, self._stress

        return self


def kruskal_stress(d_x, d_y):
    """Calculates Kruskal's stress.

    Parameters
    ----------
    d_x numpy matrix, the dissimilarities between the objects in data set X.
    d_y numpy matrix, the dissimilarities between the objects in data set Y.

    Returns
    -------

    Stress, float in the interval [0, 1], where 0 is the best possible fit and 1 is the worse.
    """
    return np.sqrt(np.power(d_x - d_y, 2).sum()
                   / np.power(d_x, 2).sum())
