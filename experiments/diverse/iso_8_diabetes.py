import numpy as np
from sklearn import datasets, svm
from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class DiabetesIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = 'Diabetes Isomap'
    plotting = True

    file = '../../datasets/diabetes/pima-indians-diabetes.data'

    reduction_method = 'isomap'
    reduction_params = {'k': 20}

    learning_parameters = [
        {'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10), 'kernel': ('rbf', 'sigmoid')},
    ]

    def _run(self):
        self.load_data()
        self.learn()

        for d in (6, 4, 2, 1):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        r = Retriever(self.file, delimiter=',')
        self.data, self.target = r.split_target().retrieve()
        self.original_data = self.data

        self.displayer \
            .load(self.data, self.target) \
            .save('datasets/diabetes') \
            .dispose()

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('Shape: %s' % str(self.data.shape))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    DiabetesIsomapExperiment().start()
