import numpy as np
from sklearn import datasets, svm
from experiments.base import CompleteExperiment
from manifold.infrastructure import Retriever


class DiabetesIsomapExperiment(CompleteExperiment):
    title = 'iso-diabetes'
    plotting = True
    reduction_method = 'isomap'
    reduction_params = {'k': 20}

    learning_parameters = [{
        'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10),
        'kernel': ('rbf', 'sigmoid')}]
    learning_cycle_components = (6, 4, 2, 1)

    def _load_data(self):
        r = Retriever('../../datasets/diabetes/pima-indians-diabetes.data',
            delimiter=',')
        self.data, self.target = r.split_target().retrieve()


if __name__ == '__main__':
    DiabetesIsomapExperiment().start()
