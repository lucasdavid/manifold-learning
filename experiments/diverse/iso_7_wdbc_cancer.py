import numpy as np
from experiments.base import CompleteExperiment
from manifold.infrastructure import Retriever


class BreastCancerExperiment(CompleteExperiment):
    title = 'iso-wdbc'
    plotting = True
    reduction_method = 'isomap'
    learning_parameters = [{
        'C': (1, 10, 100, 1000), 'gamma': (.01, .1, 1, 10),
        'kernel': ('rbf', 'sigmoid',)}]
    learning_cycle_components = (20, 10, 3, 2)

    def _load_data(self):
        r = Retriever('../../datasets/breast-cancer/wdbc.data', delimiter=',')
        r.split_column(0)  # Remove ids.

        # Split target from data and retrieve both.
        # Target feature is actually located in the 2nd column, but considering
        # we had the ids removed, it's now in the 1st one.
        self.data, self.target = r.split_target(0).retrieve()
        self.data = self.data.astype(float)


if __name__ == '__main__':
    BreastCancerExperiment().start()
