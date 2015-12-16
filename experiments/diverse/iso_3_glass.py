import numpy as np

from experiments.base import ReductionExperiment, LearningExperiment
from manifold.infrastructure import Retriever


class GlassIsomapExperiment(ReductionExperiment, LearningExperiment):
    title = 'iso_glass'
    plotting = True

    file = '../../datasets/glass/glass.data'

    reduction_method = 'isomap'
    reduction_params = {'k': 30}

    def _run(self):
        self.load_data()
        self.learn()

        for d in (8, 6, 4, 3,):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        r = Retriever(self.file, delimiter=',')
        # Remove ids.
        r.split_column(0)
        self.data, self.target = r.split_target().retrieve()
        self.original_data = self.data

        feature_names = ['Refractive index', 'Sodium', 'Magnesium']

        self.displayer \
            .load(self.data, self.target, axis_labels=feature_names) \
            .save('datasets/iso_glass') \
            .dispose()

        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))
        print('shape: %s' % str(self.data.shape))
        print('Correlation matrix:')
        print(np.corrcoef(self.data, rowvar=0))


if __name__ == '__main__':
    GlassIsomapExperiment().start()
