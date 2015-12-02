import time

from sklearn import datasets

from experiments.base import ReductionExperiment


class LIsomapExperiment(ReductionExperiment):
    title = 'L-Isomap Experiment'
    plotting = True

    benchmarks = (
        {'method': 'skisomap', 'samples': 1000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        {'method': 'lisomap', 'samples': 1000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        {'method': 'skisomap', 'samples': 4000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        {'method': 'lisomap', 'samples': 4000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        # {'method': 'skisomap', 'samples': 10000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        {'method': 'lisomap', 'samples': 10000, 'params': {'n_neighbors': 10, 'n_components': 2}},
        {'method': 'lisomap', 'samples': 30000, 'params': {'n_neighbors': 10, 'n_components': 2}},
    )

    def generate_data(self, samples):
        self.data, self.target = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        self.original_data = self.data

        if self.plotting:
            self.displayer.load(self.data, self.target)

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))
        print('Shape: %s' % str(self.data.shape))

    def _run(self):
        start = time.time()

        try:
            for benchmark in self.benchmarks:
                self.generate_data(benchmark['samples'])
                self.reduction_method = benchmark['method']
                self.reduction_params = benchmark['params']

                self.reduce()

        except KeyboardInterrupt:
            print('cancelled.', end=' ')

        print('Time elapsed: %.2f sec.' % (time.time() - start))

        if self.plotting:
            self.displayer.show()


if __name__ == '__main__':
    LIsomapExperiment().start()
