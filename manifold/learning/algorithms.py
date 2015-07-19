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

        unsaturated_nodes = {i: 0 for i in range(nodes)}

        for v in range(nodes):
            if v not in unsaturated_nodes:
                # Nothing to do if the current node was already saturated.
                continue

            links_count = unsaturated_nodes[v]
            candidate_neighbors = list(unsaturated_nodes.keys() - {v})
            candidate_links = [m[min(v, n)][max(v, n)] for n in candidate_neighbors]

            # Order neighbors by their link cost (min -> max).
            neighbors_ind = np.argsort(candidate_links)

            for i in neighbors_ind[:k - links_count]:
                neighbor = candidate_neighbors[i]

                _min, _max = min(v, neighbor), max(v, neighbor)

                if _min not in result:
                    result[_min] = dict()
                result[_min][_max] = m[_min][_max]

                unsaturated_nodes[neighbor] += 1

                if unsaturated_nodes[neighbor] == k:
                    del unsaturated_nodes[neighbor]

            unsaturated_nodes[v] += k - links_count
            if unsaturated_nodes[v] == k:
                del unsaturated_nodes[v]

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
        super().__init__(m=m, to_dimension=to_dimension)

    def run(self):
        to_dimension = self.data['to_dimension']
        count = len(self.data['m'])

        # Converts dictionary to matrix.
        p = np.array([[cost for cost in links.values()] for links in self.data['m'].values()]) ** 2

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
