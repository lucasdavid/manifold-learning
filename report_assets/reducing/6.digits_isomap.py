from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingDigitsExample(ReductionExample):
    title = '3. Reducing Digits with Isomap'
    plotting = True

    def _run(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target

        # Reduce with PCA
        self.reduction_method = 'isomap'
        self.reduction_params = {'n_components': 3, 'k': 7}
        self.reduce()

        self.reduction_params = {'n_components': 2, 'k': 7}
        self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    ReducingDigitsExample().start()
