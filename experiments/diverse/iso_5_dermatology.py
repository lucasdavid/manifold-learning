import numpy as np
from sklearn.preprocessing import Imputer

from experiments.base import CompleteExperiment


class DermatologyIsomapExperiment(CompleteExperiment):
    title = 'iso-dermatology'
    plotting = True
    reduction_method = 'isomap'
    reduction_params = {'k': 300, 'n_components': 2}

    feature_names = ['erythema', 'scaling', 'definite borders']

    learning_cycle_components = (20, 10, 3, 2)

    def _load_data(self):
        data = np.genfromtxt('../../datasets/dermatology/dermatology.data',
                             missing_values='?', delimiter=',')

        # Handle missing values by replacing them by the mean.
        data = Imputer().fit_transform(data)

        # Extract target feature.
        self.target = data[:, -1]
        self.data = np.delete(data, -1, axis=1)


if __name__ == '__main__':
    DermatologyIsomapExperiment().start()
