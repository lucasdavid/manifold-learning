from sklearn import datasets
from experiments.base import ReductionExperiment


class ReducingSwissRollExperiment(ReductionExperiment):
    title = 'Reducing The Swiss-roll with a PCA'

    reduction_method = 'pca'
    samples = 1000

    def _run(self):
        self.generate_data()

        for dimension in (3, 2, 1):
            self.reduction_params = {'n_components': dimension}
            self.reduce()
            self.plot_nearest_neighbors_graph()

        self.displayer.save(self.title)

    def generate_data(self):
        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.original_data = self.data
        self.displayer \
            .load(swiss_roll, swiss_roll_colors) \
            .save('datasets/swiss') \
            .dispose()


if __name__ == '__main__':
    ReducingSwissRollExperiment().start()
