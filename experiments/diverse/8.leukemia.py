from sklearn import datasets
from experiments.base import ReductionExperiment, LearningExperiment


class LeukemiaExperiment(ReductionExperiment, LearningExperiment):
    title = '9. Leukemia Example'
    plotting = True

    learning_parameters = [
        {'kernel': ('linear',), 'C': (1, 10, 100, 1000)},
        {'kernel': ('poly',), 'degree': (2, 3, 4), 'coef0': (0, .1, 1, 10)},
        {'kernel': ('rbf',), 'C': (1, 10, 100), 'gamma': (.001, .01, .1, 1, 10)},
        {'kernel': ('sigmoid',), 'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10), 'coef0': (0, .1, 1, 10)},
    ]

    def load_data(self):
        leukemia = datasets.fetch_mldata('leukemia', transpose_data=True)
        self.data, self.target = leukemia.data, leukemia.target

        self.original_data = self.data

        if self.plotting:
            self.displayer.load(self.data[:, 1:4], self.target)

        print('Shape: %s' % str(self.data.shape))
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

    def _run(self):
        self.load_data()
        self.learn()

        for d in (30, 20, 10):
            self.reduction_params = {'n_components': d, 'k': 5}

            self.reduce()
            self.learn()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    LeukemiaExperiment().start()
