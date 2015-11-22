import numpy as np

from experiments.base import LearningExperiment, ReductionExperiment
from manifold.infrastructure import Retriever


class BreastCancerExperiment(LearningExperiment, ReductionExperiment):
    title = '5.2.6 Breast-cancer Isomap Experiment'
    plotting = True

    learning_parameters = [
        {'C': (100,), 'kernel': ('linear',)},
    ]

    reduction_method = 'isomap'

    def _run(self):
        self.load_data()
        self.learn()

        for d in (3, 2):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.show()

    def load_data(self):
        r = Retriever('../../datasets/breast-cancer/wdbc.data', delimiter=',')

        # Remove ids, as they are not correlated in any way with the target feature.
        r.split_column(0)

        # Split target from data and retrieve both.
        # Target feature is actually located in the 2nd column, but considering we
        # had the ids removed, it's now in the 1st one.
        self.data, self.target = r.split_target(0).retrieve()
        self.data = self.data.astype(float)

        self.displayer.load(self.data, self.target, 'Breast-cancer')

        print('Covariance matrix:')
        print(self.data.shape)


if __name__ == '__main__':
    BreastCancerExperiment().start()
