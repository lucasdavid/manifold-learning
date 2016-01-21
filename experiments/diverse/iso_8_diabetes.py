import matplotlib.pyplot as plt

from experiments.base import CompleteExperiment
from manifold.infrastructure import Retriever


class DiabetesIsomapExperiment(CompleteExperiment):
    title = 'iso-diabetes'
    plotting = True
    reduction_method = 'isomap'
    reduction_params = {'k': 20}

    learning_parameters = [
        {'C': (1,), 'kernel': ('linear',)},
        {'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10),
         'kernel': ('rbf', 'sigmoid')}]
    learning_cycle_components = (6, 4, 2,)

    def _load_data(self):
        r = Retriever('../../datasets/diabetes/pima-indians-diabetes.data',
                      delimiter=',')
        self.data, self.target = r.split_target().retrieve()
        self.feature_names = ['Number of times pregnant',
                              'Plasma glucose concentration',
                              'Diastolic blood pressure (mm Hg)']

        self.displayer.colors = [plt.cm.viridis]


if __name__ == '__main__':
    DiabetesIsomapExperiment().start()
