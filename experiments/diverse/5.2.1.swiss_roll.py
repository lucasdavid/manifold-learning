import numpy as np
from sklearn import datasets, svm

from experiments.base import ReductionExperiment, LearningExperiment


class SwissRollExperiment(ReductionExperiment, LearningExperiment):
    title = '5.2.1. Swiss-roll Experiment'
    plotting = True

    samples = 1000

    reduction_method = 'isomap'
    reduction_params = {'k': 7}

    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _run(self):
        self.load_data()
        self.learn()

        for d in (2, 1):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.show()

    def load_data(self):
        self.data, self.target = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)
        self.original_data = self.data

        self.displayer.load(self.data, self.target, title='Swiss-roll')

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    SwissRollExperiment().start()
