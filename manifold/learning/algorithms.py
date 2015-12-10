import abc
from sklearn import neighbors
import numpy as np
import networkx as nx
import time
import sklearn.manifold
from scipy.spatial import distance
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils import random
from sklearn.utils.graph_shortest_path import graph_shortest_path
from ..infrastructure.base import Task, EuclideanDistancesFromDataSet, Reducer, kruskal_stress


class INearestNeighbors(Task, metaclass=abc.ABCMeta):
    def __init__(self, distance_matrix, alpha, verbose=False):
        """Performs Nearest Neighbors search in a given graph.

        Parameters
        ----------
        distance_matrix
        alpha
        verbose
        """
        assert distance_matrix is not None
        assert alpha > 0

        super().__init__(m=distance_matrix, a=alpha,
                         verbose=verbose, copying=False)


class ENearestNeighbors(INearestNeighbors):
    def run(self):
        start = time.time()

        m = self.data['m']
        e = self.data['a']

        for v in range(len(m)):
            links = m[v]

            for neighbor in range(v + 1, v + 1 + len(links)):
                if links[neighbor] > e:
                    del links[neighbor]

        if self.verbose:
            print('ENearestNeighbors took %.2f s.' % (time.time() - start))
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
        start = time.time()

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

        if self.verbose:
            print('KNearestNeighbors took %.2f s.' % (time.time() - start))
        return result


class IShortestPathFinder(Task, metaclass=abc.ABCMeta):
    def __init__(self, distance_upper_matrix, verbose=False):
        """Finds the shortest path between all nodes in a given graph.

        Parameters
        ----------
        distance_upper_matrix
            Adjacency matrix weights that represents the graph.

        verbose
            Flag indicating if info should be outputted.
        """
        g = nx.Graph()

        for v, edges in distance_upper_matrix.items():
            g.add_weighted_edges_from(((v, n, cost) for n, cost in edges.items()))

        super().__init__(g=g, verbose=verbose, copying=False)


class AllPairsDijkstra(IShortestPathFinder):
    def run(self):
        start = time.time()

        answer = nx.all_pairs_dijkstra_path_length(self.data['g'])

        if self.verbose:
            print('AllPairsDijkstra took %.2f s.' % (time.time() - start))
        return answer


class FloydWarshall(IShortestPathFinder):
    def run(self):
        start = time.time()

        answer = nx.floyd_warshall(self.data['g'])

        if self.verbose:
            print('FloydWarshall took %.2f s.' % (time.time() - start))
        return answer


class MDS(Reducer):
    def __init__(self, n_components=3, verbose=False):
        """Constructs a Multidimensional Scaling task.

        :param n_components: :int number of eigenvalues kept during the dimensionality reduction step, or the string
        'auto', which will reduce all eigenvalues that are 0.
        """
        assert isinstance(n_components, int) and n_components > 0

        super().__init__(n_components=n_components, verbose=verbose, copying=False)

        self.eigenvalues_ = self.eigenvectors_ = None

    def run(self):
        start = time.time()

        n_components = self.data['n_components']
        m = self.data['data']
        count = len(m)

        # Converts dictionary to matrix.
        # Power it by two and multiply by -1/2.
        p = m ** 2
        p *= -0.5

        # Find J = I - (1/n) * 11'
        j = np.identity(count) - (1 / count) * np.ones((count, count))

        # Find the eigenvalues (w) and eigenvectors (v) of J*P*J.
        w, v = np.linalg.eig(np.dot(np.dot(j, p), j))
        w, v = np.real(w), np.real(v)

        del p, j

        # Find set of permutations necessary to order w and v in such way that
        # w[i] <= w[i + 1], for each i in [0, len(w)).
        permutations = w.argsort()[::-1]

        # Nullify all negative eigenvalues.
        w[w < 0] = 0

        # Sort arrays and select only the first n_components values,
        # as only they are necessary to instantiate a space S such that dim(S) is :n_components.
        self.eigenvalues_ = np.sqrt(w[permutations][:n_components])
        self.eigenvectors_ = v[:, permutations][:, :n_components]

        del w, v, permutations

        # Return v * w ^ (1/2), the list of components (x, y, ...),
        # len = :n_components, corresponding to each sample in the data set.
        self.embedding = self.eigenvectors_ * self.eigenvalues_

        if self.verbose:
            print('MDS took %.2f s.' % (time.time() - start))
        return self.embedding

    def transform(self, data):
        return self.transform_dissimilarities(
            distance.squareform(
                distance.pdist(data)))

    def transform_dissimilarities(self, d):
        self.store(data=d).run()
        return self.embedding

    @property
    def stress(self):
        """Calculates Kruskal's stress.

        Returns
        -------

        Stress, float in the interval [0, 1], where 0 is the best possible fit and 1 is the worse.
        """
        assert self.embedding is not None, 'Cannot calculate stress of invalid embedding.'

        if self._stress is None:
            d_x = self.data['data']

            self._stress = kruskal_stress(d_x, distance.squareform(distance.pdist(self.embedding)))

        return self._stress


class Isomap(Reducer):
    def __init__(self,
                 nearest_method='auto', k=10, e=None,
                 n_components=3,
                 shortest_path_method='d',
                 verbose=False, copying=False):

        """Performs dimensionality reduction over a data set d
        using the Isomap algorithm.

        Parameters
        ----------
        nearest_method
        k
        e
        n_components
        shortest_path_method
        verbose
        copying
        """
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
            n_components=n_components,
            nearest_method=nearest_method,
            shortest_path_method=shortest_path_method,
            copying=copying
        )

        self._mds = None
        self._nearest_neighbors = None

    def run(self):
        start = time.time()

        m = self.data['data']
        k = self.data['k']
        e = self.data['e']

        instances_count = m.shape[0]

        m = EuclideanDistancesFromDataSet(m, verbose=self.verbose).run()

        self._nearest_neighbors = self.data['nearest_method'] == 'k' and \
                                  KNearestNeighbors(m, k, verbose=self.verbose).run() or \
                                  ENearestNeighbors(m, e, verbose=self.verbose).run()

        m = self.data['shortest_path_method'] == 'fw' and \
            FloydWarshall(self._nearest_neighbors, verbose=self.verbose).run() or \
            AllPairsDijkstra(self._nearest_neighbors, verbose=self.verbose).run()

        # Create distance matrix from neighbor map.
        dissimilarities = np.zeros((instances_count, instances_count))

        for node, edges in m.items():
            for neighbor, dissimilarity in edges.items():
                dissimilarities[node, neighbor] = dissimilarity

        del m

        self._mds = MDS(n_components=self.data['n_components'], verbose=self.verbose)
        self._mds.transform_dissimilarities(dissimilarities)
        self.embedding = self._mds.embedding

        if self.verbose:
            print('Isomap took %.2f s.' % (time.time() - start))
        return self.embedding

    @property
    def stress(self):
        """Calculates Isomap's stress according to L Shi and J. Gu's paper
        "A Fast Manifold Learning Algorithm".

        Returns
        -------

        Stress, float in the interval [0, 1], where 0 is the best possible fit and 1 is the worse.
        """
        assert self.embedding is not None, 'Cannot calculate stress from invalid embedding.'

        if self._stress is None:
            embedding_dissimilarities = EuclideanDistancesFromDataSet(self.embedding).run()

            d, d2 = 0, 0

            for node, edges in self._nearest_neighbors.items():
                for neighbor, dissimilarity in edges.items():
                    d += np.power(dissimilarity - embedding_dissimilarities[node][neighbor], 2)
                    d2 += np.power(dissimilarity, 2)

            self._stress = np.sqrt(d / d2)

        return self._stress

    def dispose(self):
        super().dispose()
        del self._mds, self._nearest_neighbors

        return self
