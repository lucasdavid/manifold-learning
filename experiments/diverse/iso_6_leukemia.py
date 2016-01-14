from sklearn import datasets, neighbors
from experiments.base import CompleteExperiment


class LeukemiaExperiment(CompleteExperiment):
    title = 'iso-leukemia'
    plotting = True
    reduction_method = 'isomap'
    reduction_params = {'n_components': 0, 'k': 5}

    knn = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

    learning_parameters = [
        {'kernel': ('linear',), 'C': (1, 10, 100, 1000)},
        {'kernel': ('poly',), 'degree': (2, 3, 4), 'coef0': (0, .1, 1, 10)},
        {'kernel': ('rbf',), 'C': (1, 10, 100),
         'gamma': (.001, .01, .1, 1, 10)},
        {'kernel': ('sigmoid',), 'C': (1, 10, 100, 1000),
         'gamma': (.001, .01, .1, 1, 10), 'coef0': (0, .1, 1, 10)},
    ]

    learning_cycle_components = (30, 20, 10)

    def _load_data(self):
        leukemia = datasets.fetch_mldata('leukemia', transpose_data=True)
        self.data, self.target = leukemia.data, leukemia.target

        self.feature_names = ['A', 'B', 'C']


if __name__ == '__main__':
    LeukemiaExperiment().start()
