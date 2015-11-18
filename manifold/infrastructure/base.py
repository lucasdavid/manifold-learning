import abc
import copy

import time
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


class EuclideanDistancesFromDataSet(Task):
    def __init__(self, data_set):
        """Creates a upper triangular matrix of distances between each sample of a given data set.

        :param data_set: the data-set that contains the points used in the distance finding.
        :return the constructed map, such that
                {i: {j:d}}, for each j e [i+1, samples)
                            for each i e [0, samples)
        """
        super().__init__(data_set=data_set, copying=False)

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

        print('Task EuclideanDistancesFromDataSet took %.2f s.' % (time.time() - start))
        return distances


class Reducer(Task, metaclass=abc.ABCMeta):
    def transform(self, data):
        self.store(data=data)
        return self.run()
