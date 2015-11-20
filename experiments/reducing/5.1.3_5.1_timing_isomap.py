import time

from sklearn import datasets

from experiments.base import ReductionExperiment


class TimingIsomapExperiment(ReductionExperiment):
    title = 'Timing Isomap Experiment'
    plotting = True

    samples = 1000

    reduction_method = 'isomap'
    reduction_params = {'n_components': 3, 'k': 7}

    def generate_data(self):
        digits = datasets.load_digits()
        self.data, self.target = digits.data, digits.target
        self.original_data = self.data

        if self.plotting:
            self.displayer.load(self.data, self.target)

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))
        print('shape: %s' % str(self.data.shape))

    def _run(self):
        self.generate_data()

        try:
            start = time.time()
            self.reduce()

        except KeyboardInterrupt:
            print('%.2f s spent in this last iteration. ' % (time.time() - start()))

        if self.plotting:
            self.displayer.show()


if __name__ == '__main__':
    TimingIsomapExperiment().start()
