from sklearn import datasets

from experiments.base import CompleteExperiment


class IrisExperiment(CompleteExperiment):
    title = 'pca-iris'
    plotting = True
    reduction_method = 'pca'
    reduction_params = {}
    reduction_params = {'n_components': 0}

    def _load_data(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target
        self.feature_names = iris.feature_names[:3]


if __name__ == '__main__':
    IrisExperiment().start()
