import abc
import numpy as np
import networkx as nx

from ..infrastructure.base import Task, EuclideanDistancesFromDataSet


class INearestNeighbors(Task, metaclass=abc.ABCMeta):
    def __init__(self, distance_matrix, alpha):
        assert distance_matrix is not None
        assert alpha > 0

        super().__init__(m=distance_matrix, a=alpha)


class ENearestNeighbors(INearestNeighbors):
    def run(self):
        m = self.data['m']
        e = self.data['a']

        for v in range(len(m)):
            links = m[v]

            for neighbor in range(v + 1, v + 1 + len(links)):
                if links[neighbor] > e:
                    del links[neighbor]

        return m


class KNearestNeighbors(INearestNeighbors):
    """Executes the K-Nearest Neighbors algorithm over a given distance matrix.

    Parameters:
        @distance_matrix: the iterable structure that contains the euclidean distances
        between the vertices.
        @a: the @k number of neighbors to be considered.

    Returns:
        The updated @distance_matrix.
    """

    def run(self):
        m, k = self.data['m'], self.data['a']
        nodes = len(m) + 1

        result = dict()

        for v in range(nodes):
            candidate_links = [m[n][v] for n in range(0, v)] + [0] + [m[v][n] for n in range(v + 1, nodes)]

            # Order neighbors by their link cost (min -> max).
            # Finally, remove :v from :indexes and select the first K neighbors.
            closest_neighbors = np.argsort(candidate_links)
            closest_neighbors = closest_neighbors[closest_neighbors != v][:k]

            result[v] = dict()

            for neighbor in closest_neighbors:
                _min, _max = min(v, neighbor), max(v, neighbor)
                result[_min][_max] = m[_min][_max]

        return result


class IShortestPathFinder(Task, metaclass=abc.ABCMeta):
    def __init__(self, distance_upper_matrix):
        g = nx.Graph()

        for v, links in distance_upper_matrix.items():
            g.add_weighted_edges_from([(v, n, link) for n, link in links.items()])

        super().__init__(g=g)


class AllPairsDijkstra(IShortestPathFinder):
    def run(self):
        return nx.all_pairs_dijkstra_path_length(self.data['g'])


class FloydWarshall(IShortestPathFinder):
    def run(self):
        return nx.floyd_warshall(self.data['g'])


class MDS(Task):
    def __init__(self, m, to_dimension=3):
        """Constructs a Multidimensional Scaling task.

        :param m: :numpy array that represents the shortest-path distances between the graph's nodes.
        :param to_dimension: :int number of eigenvalues kept during the dimensionality reduction step.
        """
        super().__init__(m=m, to_dimension=to_dimension)

    def run(self):
        to_dimension = self.data['to_dimension']
        m = self.data['m']
        count = len(m)

        # Converts dictionary to matrix.
        # Power it by two and multiply by -1/2.
        p = m ** 2
        p *= -0.5

        # Find J = I - (1/n) * 11'
        j = np.identity(count) - (1 / count) * np.ones((count, count))

        # Find the eigenvalues (w) and eigenvectors (v) of J*P*J.
        w, v = np.linalg.eig(np.dot(np.dot(j, p), j))

        del p, j

        # Find set of permutations necessary to order w and v in such way that
        # w[i] <= w[i + 1], for each i in {0, len(w) -1}.
        permutations = w.argsort()[::-1]

        # Nullify all negative eigenvalues.
        w = w.clip(min=0)

        # Sort arrays and select only the first to_dimension values,
        # as only they are necessary to instantiate a space S such that dim(S) is :to_dimension.
        w = w[permutations][:to_dimension]
        v = v[:, permutations][:, :to_dimension]

        # Return v * w ^ (1/2), the list of components (x, y, ...),
        # len = :to_dimension, corresponding to each sample in the data set.
        return np.dot(v, np.sqrt(np.diag(w)))


class Isomap(Task):
    def __init__(self,
                 data_set,
                 nearest_method='auto', k=None, e=None,
                 to_dimension=3,
                 shortest_path_method='d'):

        # Shortest-path-method must be d: dijkstra or fw: floyd-warshall.
        assert shortest_path_method == 'd' or shortest_path_method == 'fw'

        assert nearest_method == 'auto' or nearest_method == 'k' or nearest_method == 'e'
        assert k is not None or e is not None

        if nearest_method == 'auto':
            # Detects Nearest-Neighbor method to use based on which parameters were passed.
            nearest_method = 'k' if k is not None else 'e'
        else:
            # A nearest-neighbors method was chosen. Check if its parameters was passed correctly.
            assert nearest_method == 'k' and k is not None or nearest_method == 'e' and e is not None

        super().__init__(
            k=k,
            e=e,
            to_dimension=to_dimension,
            data_set=data_set,
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
        instances_count = len(data_set)

        m = EuclideanDistancesFromDataSet(data_set).run()
        m = nearest_method == 'k' and KNearestNeighbors(m, k).run() or ENearestNeighbors(m, e).run()
        m = shortest_path_method == 'fw' and FloydWarshall(m).run() or AllPairsDijkstra(m).run()

        # Create distance matrix from neighbor map.
        distance_matrix = np.zeros((instances_count, instances_count))

        for node, links in m.items():
            for neighbor, distance in links.items():
                distance_matrix[node, neighbor] = distance

        return MDS(distance_matrix, to_dimension=to_dimension).run()
