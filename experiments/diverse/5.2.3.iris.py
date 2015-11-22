import numpy as np
from sklearn import datasets, svm

from experiments.base import ReductionExperiment, LearningExperiment


class IrisIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = '5.2.3. Iris Isomap Experiment'
    plotting = True

    samples = 1000

    reduction_method = 'isomap'
    reduction_params = {'k': 7}

    def _run(self):
        self.load_data()
        self.learn()

        for d in (3, 2, 1):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        iris = datasets.load_iris()
        self.original_data = self.data = iris.data
        self.target = iris.target

        self.displayer.load(self.data, self.target)

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    IrisIsomapExperiment().start()
