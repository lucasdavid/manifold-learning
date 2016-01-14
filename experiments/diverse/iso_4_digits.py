from sklearn import datasets

from experiments.base import CompleteExperiment


class DigitsIsomapExperiment(CompleteExperiment):
    title = 'iso-digits'
    plotting = True

    reduction_method = 'isomap'
    # reduction_params = {'k': 4}

    learning_parameters = [
        {'C': (1, 10, 100), 'gamma': (.01, .1, 1), 'kernel': ('rbf', 'sigmoid')}
    ]

    learning_cycle_components = (10, 3, 2, 1)

    def _load_data(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target
        self.feature_names = ['Pixel 1', 'Pixel 2', 'Pixel 3']


if __name__ == '__main__':
    DigitsIsomapExperiment().start()
