from sklearn import datasets

from experiments.base import ReductionExperiment, LearningExperiment


class DigitsIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = 'Digits Isomap'
    plotting = True

    reduction_method = 'skisomap'
    reduction_params = {'n_neighbors': 7}

    def _run(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target
        self.original_data = self.data

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))

        self.learn()

        for dimensions in (10, 3, 2, 1):
            self.reduction_params['n_components'] = dimensions
            self.reduce()
            self.learn()
            self.data = digits.data

        if self.plotting:
            self.displayer.show()


if __name__ == '__main__':
    DigitsIsomapExperiment().start()
