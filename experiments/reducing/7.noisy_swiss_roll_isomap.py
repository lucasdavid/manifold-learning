import networkx as nx
import pylab as plt
from sklearn import datasets

from experiments.base import ReductionExperiment
from manifold.learning import algorithms


class NoisySwissRollIsomapExperiment(ReductionExperiment):
    title = '7. Noisy Swiss-roll Isomap example'

    plotting = True

    samples = 1000
    noise_range = (.8, .8)
    noise_increment = .2
    neighbors = (7,)
    reduction_method = 'isomap'

    def generate_data_set(self, noise):
        self.data, self.target = datasets.make_swiss_roll(
            n_samples=self.samples, random_state=0, noise=noise)

        self.original_data = self.data

        self.displayer \
            .load(self.data, self.target, title='Swiss-roll (noise: %.2f)' % noise) \
            .render() \
            .dispose()

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

    def _run(self):
        noise = self.noise_range[0]

        while noise <= self.noise_range[1]:
            self.generate_data_set(noise)

            for n in self.neighbors:
                self.reduction_params = {'n_components': 2, 'k': n}
                self.reduce()

            noise += self.noise_increment

            if self.plotting:
                self.displayer.aspect = (10, 70)
                self.displayer.render().dispose()
                self.draw_nearest_neighbor_graph_found()

    def draw_nearest_neighbor_graph_found(self):
        d = algorithms.EuclideanDistancesFromDataSet(self.original_data).run()
        e = algorithms.KNearestNeighbors(d, alpha=10).run()
        g = nx.Graph(e)
        del d, e

        pos = {n: location[:2] for n, location in enumerate(self.data)}

        nx.draw(g, pos=pos, node_size=40, node_color=self.target, alpha=.6,
                width=1, edge_color='#cccccc', with_labels=False)

        plt.show()


if __name__ == '__main__':
    NoisySwissRollIsomapExperiment().start()
