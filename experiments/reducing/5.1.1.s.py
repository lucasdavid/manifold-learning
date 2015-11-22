from sklearn import datasets

from experiments.base import ReductionExperiment


class SManifoldExperiment(ReductionExperiment):
    title = 'S Experiment'
    plotting = True

    reduction_method = 'skisomap'
    reduction_params = {'n_components': 2, 'n_neighbors': 10}

    def load_data(self):
        self.data, self.target = datasets.make_s_curve(n_samples=1000)
        self.original_data = self.data

        self.displayer.load(self.data, self.target)
        self.displayer.aspect = (20, -30)

    def _run(self):
        self.load_data()

        self.reduce()

        self.plot_nearest_neighbors_graph(position='reduced')
        self.displayer.show()


if __name__ == '__main__':
    SManifoldExperiment().start()
