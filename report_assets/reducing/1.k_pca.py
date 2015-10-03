import numpy as np
from sklearn import preprocessing

from report_assets.base import ReductionExample


class ReducingKExample(ReductionExample):
    title = '1. Reducing K Data set'

    def _run(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1)

        print('Covariance of K:')
        print(np.cov(self.data, rowvar=0))
        print('Data size: %i' % self.data.nbytes)

        self.method = 'pca'
        self.params = {'n_components': 2}
        self.reduce()

        self.params = {'n_components': 1}
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingKExample().start()
