from sklearn import datasets

from experiments.base import ReductionExperiment


class ReducingIrisExperiment(ReductionExperiment):
    title = '2. Reducing Iris flower'

    def _run(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target

        # Reduce with PCA
        self.reduction_method = 'pca'
        self.reduction_params = {'n_components': 2}
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingIrisExperiment().start()
