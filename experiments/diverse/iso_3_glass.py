import numpy as np

from manifold.infrastructure import Retriever
from experiments.base import CompleteExperiment


class GlassIsomapExperiment(CompleteExperiment):
    title = 'iso-glass'
    plotting = True
    reduction_method = 'isomap'
    reduction_params = {'k': 30}

    learning_cycle_components = (8, 6, 4, 3,)

    file = '../../datasets/glass/glass.data'
    feature_names = ['Refractive index', 'Sodium', 'Magnesium']

    def _load_data(self):
        r = Retriever(self.file, delimiter=',')
        r.split_column(0)  # Remove ids.
        self.data, self.target = r.split_target().retrieve()
        self.original_data = self.data


if __name__ == '__main__':
    GlassIsomapExperiment().start()
