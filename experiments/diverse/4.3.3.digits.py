from sklearn import datasets

from experiments.base import ReductionExperiment, LearningExperiment


class DigitsExperiment(ReductionExperiment, LearningExperiment):
    title = '4.3.3. Digits'
    plotting = True

    def _run(self):
        self.load_data()
        self.learn()

        self.reduction_method = 'pca'

        for dimensions in (10, 3, 2, 1):
            self.reduction_params = {'n_components': dimensions}
            self.reduce()
            self.learn()

        self.displayer.show()

    def load_data(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target
        self.original_data = self.data

        self.displayer.load(self.data, self.target)

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    DigitsExperiment().start()
