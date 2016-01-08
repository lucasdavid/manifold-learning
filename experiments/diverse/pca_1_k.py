import numpy as np
from sklearn import preprocessing, neighbors
from experiments.base import CompleteExperiment


class KExperiment(CompleteExperiment):
    title = 'pca-k'
    plotting = True
    reduction_method = 'pca'
    knn = neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=-1)

    displaying_cycle_components = (2, 1)
    learning_cycle_components = (2, 1)

    labels = ['A', 'B']

    def _load_data(self):
        np.random.seed(0)
        mean, cov, n = [0, 0], [[1, 1], [1.4, 1.5]], 1000

        self.data = np.random.multivariate_normal(mean, cov, n)
        self.data = preprocessing.scale(self.data)
        self.target = self.data.sum(axis=1).astype(int)

        print('Correlation of K')
        print(np.corrcoef(self.data, rowvar=0))
        print('Data size: %.2f KB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    KExperiment().start()
