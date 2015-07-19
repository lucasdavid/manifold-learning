import numpy as np
import networkx as nx

from ..infrastructure.base import Task, EuclideanDistancesFromDataSet


class ENearestNeighbors(Task):
    def __init__(self, distance_matrix, e):
        super().__init__(m=distance_matrix, e=e)

    def run(self):
        m = self.data['m']
        e = self.data['e']

        for v in range(len(m)):
            links = m[v]

            for neighbor in range(v + 1, v + 1 + len(links)):
                if links[neighbor] > e:
                    del links[neighbor]

        return m


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
        nodes = len(m) + 1

        for v in range(nodes):
            links_established_count = len([_ for previous in range(v) if v in m[previous]])

            if links_established_count >= k:
                # Nothing to do if this node already has K neighbors.
                continue

            candidate_links = [m[neighbor][v] for neighbor in range(v + 1, nodes)]

            # Order neighbors by their link cost (min -> max).
            neighbors = np.argsort(candidate_links) + v + 1
            # Select neighbors that are not between the first K -links_established_count less costly links.
            neighbors = neighbors[k - links_established_count:]

            for n in neighbors:
                # Remove all neighbors selected on the previous step.
                del m[v][n]

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
        count = len(self.data['m'])

        # Converts dictionary to matrix.
        p = np.array([[cost for cost in links.values()] for links in self.data['m'].values()])  ** 2

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
    def __init__(self,
                 data_set, color,
                 nearest_method='k', k=4, e=20,
                 to_dimension=3,
                 shortest_path_method='dijkstra'):

        assert shortest_path_method == 'dijkstra' or shortest_path_method == 'floyd-warshall'
        assert nearest_method == 'k' or nearest_method == 'e'

        super().__init__(
            k=k,
            e=e,
            to_dimension=to_dimension,
            data_set=data_set,
            color=color,
            nearest_method=nearest_method,
            shortest_path_method=shortest_path_method,
            copying=True
        )

    def run(self):
        data_set = self.data['data_set']
        to_dimension = self.data['to_dimension']
        shortest_path_method = self.data['shortest_path_method']
        nearest_method = self.data['nearest_method']
        k = self.data['k']
        e = self.data['e']

        m = EuclideanDistancesFromDataSet(data_set).run()
        m = nearest_method == 'k' and KNearestNeighbors(m, k).run() or ENearestNeighbors(m, e).run()
        m = shortest_path_method == 'dijkstra' and FloydWarshall(m).run() or AllPairsDijkstra(m).run()

        return MDS(m, to_dimension=to_dimension).run()
