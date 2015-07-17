import abc
import copy
import numpy as np
import sys


class Algorithm(object, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.data = None
        self.load(**kwargs)

    def load(self, **kwargs):
        self.data = copy.deepcopy(kwargs)

    def execute(self):
        raise NotImplementedError


class FloydWarshall(Algorithm):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix=distance_matrix)

    def execute(self):
        assert self.data['distance_matrix'] is not None

        m = self.data['distance_matrix']
        count = len(m)

        max = np.max(m)

        V = []
        for i in range(count):
            V.append([m[i][j] or max + 1 for j in range(count)])
            V[i][i] = 0

        V = np.array(V)

        for k in range(count):
            for i in range(count):
                for j in range(count):
                    if V[i][j] > V[i][k] + V[k][j]:
                        V[i][j] = V[i][k] + V[k][j]

        return V


class Isomap(Algorithm):
    def __init__(self, proximity_matrix, to_dimension=3):
        super().__init__(proximity_matrix=proximity_matrix, to_dimension=to_dimension)

    def execute(self):
        assert self.data['proximity_matrix'] is not None

        to_dimension = self.data['to_dimension']

        p = np.array(self.data['proximity_matrix'])
        p = np.power(p, 2)

        count = len(p)

        j = np.identity(count) - (1 / count) * np.ones((count, count))
        b = (-1 / 2) * np.dot(np.dot(j, p), j)

        w, v = np.linalg.eig(b)

        # Find set of permutations necessary to order w and v in such way that
        # w[i] <= w[i + 1], for each i in {0, len(w) -1}.
        permutations = w.argsort()[::-1]

        # Sort arrays and select only the first to_dimension values,
        # as only they are necessary to instantiate a space S such that dim(S) == to_dimension.
        w = w[permutations][:to_dimension]
        v = v[:, permutations][:, :to_dimension]

        return np.dot(v, np.sqrt(np.diag(w)))
