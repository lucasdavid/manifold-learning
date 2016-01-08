from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import Imputer
from manifold.infrastructure import Retriever
from experiments.base import CompleteExperiment


class UnknownExperiment(CompleteExperiment):
    title = 'iso-unknown'
    reduction_method = 'isomap'
    reduction_params = {'n_components': 3}

    plotting = True

    learning_parameters = [
        {'C': (1, 10, 100, 1000), 'kernel': ('linear',)},
        # {'C': (1, 10,), 'gamma': (.01, .1, 1,), 'kernel': ('rbf',)},
    ]

    learning_cycle_components = (75, 50, 25)

    def load_data(self):
        self.data, self.target = load_svmlight_file(
            '../../datasets/unknown/dt3.trn.svm', zero_based=True)
        self.original_data = self.data = self.data.toarray()
        self.data_tst, self.target_tst = load_svmlight_file(
            '../../datasets/unknown/dt3.tst.svm', zero_based=True)


if __name__ == '__main__':
    UnknownExperiment().start()
