from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import Imputer
from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class UnknownExperiment(ReductionExperiment, LearningExperiment):
    title = 'Unknown Example'
    data_file = '../../datasets/unknown/dt3.%s.svm'

    reduction_method = 'pca'
    reduction_params = {'n_components': 3}

    plotting = True

    learning_parameters = [
        {'C': (1, 10, 100, 1000), 'kernel': ('linear',)},
        # {'C': (1, 10,), 'gamma': (.01, .1, 1,), 'kernel': ('rbf',)},
    ]

    def load_data(self):
        self.data, self.target = load_svmlight_file(self.data_file % 'trn', zero_based=True)
        self.original_data = self.data = self.data.toarray()
        self.data_tst, self.target_tst = load_svmlight_file(self.data_file % 'tst', zero_based=True)

        self.displayer.load(self.data, self.target)

        print('Shape: %s' % str(self.data.shape))
        print('Data size: %s, %.2f KB.' % (self.data.shape, (self.data.nbytes / 1024)))

    def _run(self):
        self.load_data()

        self.reduction_method = 'skisomap'
        self.reduction_params = {
            'n_neighbors': 10,
            'n_components': 50,
        }

        self.reduce()
        self.learn()


if __name__ == '__main__':
    UnknownExperiment().start()
