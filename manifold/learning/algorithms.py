import numpy as np
import networkx as nx

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

        for node in range(nodes):
            links_to_neighbors = [m[neighbor][node] for neighbor in range(node)] \
                + [0] \
                + [m[neighbor][node] for neighbor in range(node + 1, nodes)]

            # Order neighbors by their link cost (min -> max).
            neighbors = np.argsort(links_to_neighbors)
            # Select neighbors that are not the current node and are not between the first K less costly links.
            neighbors = neighbors[neighbors != node][:k]

            for neighbor in neighbors:
                if node in m:
                    if neighbor in m[node]:
                        del m[node][neighbor]
                elif neighbor in m:
                    if node in m[neighbor]:
                        del m[neighbor][node]

        return m


class AllPairsDijkstra(Task):
    def __init__(self, distance_upper_tri_matrix):
        super().__init__(m=distance_upper_tri_matrix)

    def run(self):
        m = self.data['m']
        return nx.all_pairs_dijkstra_path_length(
            nx.Graph(m))


class FloydWarshall(Task):
    def __init__(self, distance_matrix):
        super().__init__(m=distance_matrix)

    def run(self):
        m = self.data['m']
        return nx.floyd_warshall(
            nx.Graph(m))


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
    def __init__(self, data_set, color, k=4, to_dimension=3, method='dijkstra'):
        assert method == 'dijkstra' or method == 'floyd-warshall'

        super().__init__(
            k=k,
            to_dimension=to_dimension,
            data_set=data_set,
            color=color,
            method=method,
            copying=True
        )

    def run(self):
        data_set = self.data['data_set']
        k = self.data['k']
        to_dimension = self.data['to_dimension']
        method = self.data['method']

        m = EuclideanDistancesFromDataSet(data_set).run()
        m = KNearestNeighbors(m, k).run()
        m = method == 'dijkstra' and FloydWarshall(m).run() or AllPairsDijkstra(m).run()
        return MDS(m, to_dimension=to_dimension).run()
