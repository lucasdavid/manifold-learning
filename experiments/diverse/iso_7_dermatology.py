import numpy as np
from sklearn.preprocessing import Imputer

from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class DermatologyIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = '5.2.7. Dermatology Isomap Experiment'
    plotting = True

    file = '../../datasets/dermatology/dermatology.data'

    reduction_method = 'isomap'
    reduction_params = {'k': 20, 'n_components': 2}

    def _run(self):
        self.load_data()
        self.learn()

        for d in (20, 10, 3, 2):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        data = np.genfromtxt(self.file, missing_values='?', delimiter=',')

        # Handle missing values by replacing them by the mean.
        data = Imputer().fit_transform(data)

        self.target = data[:, -1]
        self.original_data = self.data = data = np.delete(data, -1, axis=1)

        self.displayer \
            .load(self.data, self.target) \
            .save('datasets/dermatology') \
            .dispose()

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('shape: %s' % str(self.data.shape))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    DermatologyIsomapExperiment().start()
