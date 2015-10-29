from sklearn import datasets

from report_assets.base import ReductionExample, LearningExample


class DigitsIsomapExample(ReductionExample, LearningExample):
    title = '7. Digits Reduced with Isomap and Learned Example'
    plotting = True

    def _run(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.learn()

        self.reduction_method = 'skisomap'

        for d in (10, 3, 2, 1):
            self.reduction_params = {'n_components': d, 'n_neighbors': 7}
            self.reduce()

            self.data = self.reduced_data
            self.learn()
            self.data = digits.data

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    DigitsIsomapExample().start()
