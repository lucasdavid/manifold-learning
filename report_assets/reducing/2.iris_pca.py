from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingIrisExample(ReductionExample):
    title = '1. Reducing Iris flower'

    def _run(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target

        # Reduce with PCA
        self.method = 'pca'
        self.params = {'n_components': 2}
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingIrisExample().start()
