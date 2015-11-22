import numpy as np
from sklearn import datasets, svm
from sklearn.preprocessing import Imputer

from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class DermatologyIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = '5.2.7. Dermatology Isomap Experiment'
    plotting = True

    file = '../../datasets/dermatology/dermatology.data'

    reduction_method = 'isomap'
    reduction_params = {'k': 7}

    def _run(self):
        self.load_data()
        self.learn()

        for d in (10, 3, 2, 1):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        r = Retriever(self.file, delimiter=',')
        r.split_column(0)
        data = np.genfromtxt(self.file, missing_values='?', delimiter=',')

        # Handle missing values by replacing them by the mean.
        data = Imputer().fit_transform(data)

        self.target = data[:, -1]
        self.original_data = self.data = data = np.delete(data, -1, axis=1)

        self.displayer.load(self.data, self.target)

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    DermatologyIsomapExperiment().start()
