from sklearn import datasets

from experiments.base import CompleteExperiment


class IrisIsomapExperiment(CompleteExperiment):
    title = 'iso-iris'
    plotting = True

    reduction_method = 'isomap'
    reduction_params = {'k': 30}

    def _load_data(self):
        iris = datasets.load_iris()
        self.data = iris.data
        self.target = iris.target
        self.feature_names = iris.feature_names[:3]


if __name__ == '__main__':
    IrisIsomapExperiment().start()
