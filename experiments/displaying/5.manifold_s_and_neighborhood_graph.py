import networkx as nx
import pylab as plt
from scipy.spatial import distance
from sklearn import datasets

from experiments.base import ReductionExperiment
from manifold.learning import algorithms


class DisplayingDatasetAsGraphExperiment(ReductionExperiment):
    title = '5. Displaying the S-Dataset as a Graph Example'
    plotting = True

    def load_data(self):
        self.data, self.target = datasets.make_s_curve(n_samples=1000)
        self.original_data = self.data

        if self.plotting:
            self.displayer.load(self.data, self.target)
            self.displayer.aspect = (20, -30)

    def _run(self):
        self.load_data()

        self.reduction_method = 'skisomap'
        self.reduction_params = {'n_neighbors': 10}

        for dimension in (1, 2):
            self.reduction_params['n_components'] = dimension
            self.reduce()

        self.draw_nearest_neighbor_graph_found()

        if self.plotting:
            self.displayer.render()

    def draw_nearest_neighbor_graph_found(self):
        d = distance.squareform(distance.pdist(self.original_data))
        e = algorithms.KNearestNeighbors(d, alpha=10).run()
        g = nx.Graph(e)
        del d, e

        pos = {n: location[:2] for n, location in enumerate(self.data)}

        nx.draw(g, pos=pos, node_size=40, node_color=self.target, alpha=.6,
                width=1, edge_color='#cccccc', with_labels=False)

        plt.show()


if __name__ == '__main__':
    DisplayingDatasetAsGraphExperiment().start()
