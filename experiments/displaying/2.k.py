import numpy as np
from sklearn import preprocessing

from experiments.base import Experiment


class DisplayingKExperiment(Experiment):
    title = '2. Displaying K data set'

    def _run(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        data = np.random.multivariate_normal(mean, cov, n)
        data = preprocessing.scale(data)
        target = data.sum(axis=1)

        self.displayer.load(data, target).render()

        print('Covariance of K')
        print(np.cov(data, rowvar=0))


if __name__ == '__main__':
    DisplayingKExperiment().start()
