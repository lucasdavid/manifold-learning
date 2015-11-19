import time

from experiments.base import ReductionExperiment
from manifold.infrastructure import Retriever


class SpamExperiment(ReductionExperiment):
    title = '7. Spam Reduced Example'
    file = '../../datasets/spam/spambase.data'
    plotting = True

    def load_data(self):
        self.data, self.target = Retriever(self.file, delimiter=',').split_target().retrieve()
        self.original_data = self.data

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))
        print('shape: %s' % str(self.data.shape))

    def _run(self):
        self.load_data()

        for m, params in (
                ('skisomap', {'n_components': 3, 'n_neighbors': 7}),
                ('isomap', {'n_components': 3, 'k': 7}),
        ):
            try:
                start = time.time()

                self.reduction_method = m
                self.reduction_params = params

                self.reduce()

            except KeyboardInterrupt:
                print('%.2f s spent in this last iteration. ' % (time.time() - start()))

        if self.plotting:
            self.displayer.show()


if __name__ == '__main__':
    SpamExperiment().start()
