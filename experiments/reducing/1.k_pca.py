import numpy as np
from sklearn import preprocessing

from report_assets.base import ReductionExample


class ReducingKExample(ReductionExample):
    title = '1. Reducing K With PCA Example'
    plotting = True

    reduction_method = 'pca'

    def _run(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1)

        print('Covariance of K:')
        print(np.cov(self.data, rowvar=0))
        print('Data size: %.1f KB\n' % (self.data.nbytes / 1024))

        for dimension in (2, 1):
            self.reduction_params = {'n_components': dimension}
            self.reduce()

            print('Covariance of reduced K:')
            print(np.cov(self.reducer.components_, rowvar=0))

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    ReducingKExample().start()
