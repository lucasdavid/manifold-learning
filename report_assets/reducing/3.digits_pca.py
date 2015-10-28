from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingDigitsExample(ReductionExample):
    title = '3. Reducing Digits'
    plotting = True

    def _run(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target

        # Reduce with PCA
        self.reduction_method = 'pca'
        self.reduction_params = {'n_components': 3}
        self.reduce()

        self.reduction_params = {'n_components': 2}
        self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    ReducingDigitsExample().start()
