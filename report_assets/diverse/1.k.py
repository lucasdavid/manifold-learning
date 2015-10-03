import numpy as np
from sklearn import preprocessing

from report_assets.base import ReductionExample, LearningExample


class KExample(ReductionExample, LearningExample):
    title = '1. K Data set'

    def _run(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1).astype(int)

        self.displayer.load(self.data, self.target)

        print('Covariance of K')
        print(np.cov(self.data, rowvar=0))
        print('Data size: %i' % self.data.nbytes)

        # Learn, through GridSearch, the data set K.
        self.learn()

        # Reduce K to only one component.
        self.method = 'pca'
        self.params = {'n_components': 2}
        self.reduce()

        self.learn()

        # Reduce K to only one component.
        self.method = 'pca'
        self.params = {'n_components': 1}
        self.reduce()

        # Learn reduced K.
        self.learn()

        self.displayer.render()


if __name__ == '__main__':
    KExample().start()
