import abc
import copy
import numpy as np

from scipy.spatial.distance import pdist


class Task(object, metaclass=abc.ABCMeta):
    def __init__(self, copying=True, **kwargs):
        self.data = {}
        self.store(copying, **kwargs)

    def store(self, copying=True, **kwargs):
        self.data.update(copying and copy.deepcopy(kwargs) or kwargs)
        return self

    def dispose(self):
        self.data = {}
        return self

    def run(self):
        raise NotImplementedError


class EuclideanDistancesFromDataSet(Task):
    def __init__(self, data_set):
        super().__init__(data_set=data_set, copying=False)

    def run(self):
        data_set = self.data['data_set']
        samples = len(data_set)

        # Find the distance between each pair of points.
        distances = pdist(data_set)

        x = np.zeros((samples, samples))

        for i in range(samples):
            for j in range(i + 1, samples):
                x[i][j] = x[j][i] = distances[0]
                distances = distances[1:]

        return x
