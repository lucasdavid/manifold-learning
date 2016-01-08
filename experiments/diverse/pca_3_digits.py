from sklearn import datasets
from experiments.base import CompleteExperiment


class DigitsExperiment(CompleteExperiment):
    title = 'pca-digits'
    plotting = True

    learning_cycle_components = (10, 3, 2, 1)

    def _load_data(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target


if __name__ == '__main__':
    DigitsExperiment().start()
