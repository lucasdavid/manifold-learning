import abc
import copy
import time
from sklearn import neighbors

import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform


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
            distances[i] = {i + n + 1: d for n, d in
                            enumerate(d[0:samples - i - 1])}
            d = d[samples - i - 1:]

        if self.verbose:
            print('Task EuclideanDistancesFromDataSet took %.2f s.' % (
                time.time() - start))
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
    """Calculates Kruskal's stress for data sets reduced with ISOMAP.

    :param d_x: the dissimilarities between the objects in data set X.
    :param d_y: the dissimilarities between the objects in data set Y.

    :return: Stress, float in the interval [0, 1], where 0 is the best possible
    fit and 1 is the worse.
    """
    return np.sqrt(np.power(d_x - d_y, 2).sum() / np.power(d_x, 2).sum())


def partial_kruskal_stress(X, Y, n_neighbors):
    """Calculates Kruskal's stress for data sets reduced with ISOMAP.

    :param X: the original data set X.
    :param Y: the data set Y (a reduction of X).
    :param n_neighbors: how many neighbors should be considered
        when performing nearest neighbor search.

    :return: Stress, float in the interval [0, 1], where 0 is the best possible
    fit and 1 is the worse.
    """
    d_x, d_y = squareform(pdist(X)), squareform(pdist(Y))

    nbrs = (neighbors.NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            .fit(X)
            .kneighbors(return_distance=False))

    rows = np.arange(X.shape[0]).reshape(X.shape[0], 1)
    d_x_omega = d_x[rows, nbrs]
    d_y_omega = d_y[rows, nbrs]

    del rows, nbrs

    return np.sqrt(np.power(d_x_omega - d_y_omega, 2).sum() /
                   np.power(d_x_omega, 2).sum())


def class_stress(X, Y, target, n_neighbors=5, n_jobs=1):
    """Calculate the Class stress for a labeled data set X and its reduction Y.

    Parameters
    ----------
    X : original data set.

    Y : the reduced data set.

    target : the target feature of the data set X.

    n_neighbors : the number of neighbors considered when building
        the nearest neighborhood graph.

    n_jobs : the number of jobs triggered for the nearest
        neighbors algorithm.
    """
    assert X.shape[0] == Y.shape[0] == target.shape[
        0], 'Number of samples do not match.'

    target += abs(target.min()) + 1

    stress_X = _class_stress(X, target)
    stress_Y = _class_stress(Y, target)

    return abs(stress_X - stress_Y)


def _class_stress(X, target):
    n_samples = X.shape[0]

    coef = distance.pdist(X)
    coef_sum = coef.sum()
    coef_mean = coef_sum / coef.shape[0]
    coef = 2 * coef_mean - coef

    stress_X_ = 0
    current = 0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            stress_X_ += (coef[current] / coef_sum) \
                         * abs(target[i] - target[j]) / (
                             abs(target[i]) + abs(target[j]))

    return stress_X_
