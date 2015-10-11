from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingDigitsExample(ReductionExample):
    title = '1. Reducing Digits'

    def _run(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target

        # Reduce with PCA
        self.method = 'pca'
        self.params = {'n_components': 3}
        self.reduce()

        self.params = {'n_components': 2}
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingDigitsExample().start()
