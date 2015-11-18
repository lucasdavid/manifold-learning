import numpy as np
from sklearn import preprocessing

from experiments.base import ReductionExperiment, LearningExperiment


class KExperiment(ReductionExperiment, LearningExperiment):
    title = '1. K PCA Example'

    def _run(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1).astype(int)

        print('Covariance of K')
        print(np.cov(self.data, rowvar=0))
        print('Data size: %i' % self.data.nbytes)

        # Learn, through GridSearch, the data set K.
        self.learn()

        # Reduce dimensions of K.
        self.reduction_method = 'pca'

        for dimension in (2, 1):
            self.reduction_params = {'n_components': dimension}
            self.reduce()

            original_data = self.data
            self.data = self.reduced_data
            self.learn()

            self.data = original_data

        self.displayer.render()


if __name__ == '__main__':
    KExperiment().start()
