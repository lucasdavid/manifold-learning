import numpy as np

from ..infrastructure.base import Task, EuclideanDistancesFromDataSet


class KNearestNeighbors(Task):
    def __init__(self, distance_matrix, k=4):
        """Executes the K-Nearest Neighbors algorithm over a given distance matrix.

        :param distance_matrix: the iterable structure that contains the euclidean distances
        between the vertices.
        :param k: the number of neighbors to be considered. Default is 4.
        """
        super().__init__(m=distance_matrix, k=k)

    def run(self):
        m, k = self.data['m'], self.data['k']
        nodes = len(m)

        x = np.zeros((nodes, nodes))

        for node in range(nodes):
            # Gets K closest neighbors, but removes node first.
            neighbors = np.array(np.argsort(m[node]))
            neighbors = neighbors[neighbors != node][:k]

            for neighbor in neighbors:
                # Fill distance matrix with these neighbors distances.
                x[node][neighbor] = m[node][neighbor]

        return x


class FloydWarshall(Task):
    def __init__(self, distance_matrix):
        super().__init__(m=distance_matrix)

    def run(self):
        m = self.data['m']
        count = len(m)

        max = np.max(m)

        V = []
        for i in range(count):
            V.append([m[i][j] or np.inf for j in range(count)])
            V[i][i] = 0

        V = np.array(V)

        for k in range(count):
            for i in range(count):
                for j in range(count):
                    V[i][j] = min(V[i][j], V[i][k] + V[k][j])

        return V


class MDS(Task):
    def __init__(self, m, to_dimension=3):
        super().__init__(m=m, to_dimension=to_dimension)

    def run(self):
        to_dimension = self.data['to_dimension']
        p = np.array(self.data['m']) ** 2

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


class Isomap(Task):
    def __init__(self, data_set, color, k=4, to_dimension=3):
        super().__init__(
            k=k,
            to_dimension=to_dimension,
            data_set=data_set,
            color=color,
            copying=False
        )

    def run(self):
        data_set = self.data['data_set']
        k = self.data['k']
        to_dimension = self.data['to_dimension']

        m = EuclideanDistancesFromDataSet(data_set).run()
        m = KNearestNeighbors(m, k).run()
        m = FloydWarshall(m).run()
        return MDS(m, to_dimension=to_dimension).run()
