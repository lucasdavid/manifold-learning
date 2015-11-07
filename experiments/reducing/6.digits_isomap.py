from sklearn import datasets

from report_assets.base import ReductionExample


class DigitsIsomapExample(ReductionExample):
    title = '7. Digits Reduced with Isomap Example'
    plotting = True

    def _run(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.reduction_method = 'isomap'

        for d in (10, 3, 2, 1):
            self.reduction_params = {'n_components': d, 'k': 7}
            self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    DigitsIsomapExample().start()
