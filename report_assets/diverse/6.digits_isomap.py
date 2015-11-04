from sklearn import datasets
from report_assets.base import ReductionExample, LearningExample


class DigitsIsomapExample(ReductionExample, LearningExample):
    title = '6. Digits Reduced with Isomap and Learned Example'
    plotting = True

    reduction_method = 'isomap'
    reduction_params = {'n_neighbors': 7}

    def _run(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.learn()

        for dimensions in (10, 3, 2, 1):
            self.reduction_params['n_components'] = dimensions
            self.reduce()

            self.data = self.reduced_data
            self.learn()
            self.data = digits.data
            del self.reduced_data

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    DigitsIsomapExample().start()
