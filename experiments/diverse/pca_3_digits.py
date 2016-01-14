from sklearn import datasets
from experiments.base import CompleteExperiment


class DigitsExperiment(CompleteExperiment):
    title = 'pca-digits'
    plotting = True

    reduction_method = 'pca'
    reduction_params = {}
    learning_cycle_components = (10, 3, 2, 1)

    def _load_data(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target
        self.feature_names = ['Pixel 1', 'Pixel 2', 'Pixel 3']


if __name__ == '__main__':
    DigitsExperiment().start()
