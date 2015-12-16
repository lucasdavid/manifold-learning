import numpy as np
from sklearn.preprocessing import Imputer

from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class DermatologyIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = 'iso_dermatology'
    plotting = True

    file = '../../datasets/dermatology/dermatology.data'

    reduction_method = 'skisomap'
    reduction_params = {'n_neighbors': 330, 'n_components': 2}

    def _run(self):
        self.load_data()
        # self.learn()

        # for d in (20, 10, 3, 2):
        for d in (3, 2):
            self.reduction_params['n_components'] = d
            self.reduce()
            # self.learn()

        # self.displayer.save(self.title)
        self.displayer.show()

    def load_data(self):
        data = np.genfromtxt(self.file, missing_values='?', delimiter=',')

        # Handle missing values by replacing them by the mean.
        data = Imputer().fit_transform(data)

        self.target = data[:, -1]
        self.original_data = self.data = data = np.delete(data, -1, axis=1)

        feature_names = ['erythema', 'scaling', 'definite borders']

        self.displayer \
            .load(self.data, self.target, axis_labels=feature_names) \
            .save('datasets/dermatology') \
            .dispose()

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('shape: %s' % str(self.data.shape))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    DermatologyIsomapExperiment().start()
