import numpy as np
from sklearn import preprocessing
from experiments.base import ReductionExperiment, LearningExperiment


class KExperiment(ReductionExperiment, LearningExperiment):
    title = '4.3.1. K Data Set'
    plotting = True

    def _run(self):
        self.generate_data()

        # Learn, through GridSearch, the data set K.
        self.learn()

        # Reduce dimensions of K.
        self.reduction_method = 'pca'

        for dimension in (2, 1):
            self.reduction_params = {'n_components': dimension}
            self.reduce()
            self.learn()

        if self.plotting:
            self.displayer.render()

    def generate_data(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.original_data = self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1).astype(int)

        if self.plotting:
            self.displayer.load(self.data, self.target)

        print('Correlation of K')
        print(np.corrcoef(self.data, rowvar=0))
        print('Data size: %.2f KB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    KExperiment().start()
