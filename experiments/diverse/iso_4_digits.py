from sklearn import datasets

from experiments.base import ReductionExperiment, LearningExperiment


class DigitsIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = 'iso_digits'
    plotting = True

    reduction_method = 'isomap'
    reduction_params = {'k': 4}

    def _run(self):
        self.load_data()
        self.evaluate()
        self.learn()

        for dimensions in (10, 3, 2, 1):
            self.reduction_params['n_components'] = dimensions
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target
        self.original_data = self.data

        self.displayer \
            .load(self.data[:, 1:4], self.target, axis_labels=['A', 'B', 'C']) \
            .save('datasets/iso_digits') \
            .dispose()

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    DigitsIsomapExperiment().start()
