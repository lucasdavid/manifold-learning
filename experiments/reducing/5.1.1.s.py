import networkx as nx
import pylab as plt
from sklearn import datasets

from experiments.base import ReductionExperiment
from manifold.learning import algorithms


class SManifoldExperiment(ReductionExperiment):
    title = '5.1.1. S Experiment'
    plotting = True

    reduction_method = 'isomap'
    reduction_params = {'n_components': 2, 'k': 10}

    def load_data(self):
        self.data, self.target = datasets.make_s_curve(n_samples=1000)
        self.original_data = self.data

        self.displayer.load(self.data, self.target)
        self.displayer.aspect = (20, -30)

    def _run(self):
        self.load_data()

        self.reduce()

        self.draw_nearest_neighbor_graph_found()
        self.displayer.show()

    def draw_nearest_neighbor_graph_found(self):
        """Draw Nearest Neighbor Graph.

        This method must run after .reduce(), as it will use the
        vertices' positions found by the MDS method.

        """
        d = algorithms.EuclideanDistancesFromDataSet(self.original_data).run()
        e = algorithms.KNearestNeighbors(d, alpha=10).run()
        g = nx.Graph(e)
        del d, e

        pos = {n: location[:2] for n, location in enumerate(self.data)}

        nx.draw(g, pos=pos, node_size=40, node_color=self.target, alpha=.6,
                width=1, edge_color='#cccccc', with_labels=False)

        plt.show()


if __name__ == '__main__':
    SManifoldExperiment().start()
