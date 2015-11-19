import numpy as np
from sklearn import preprocessing, datasets

from experiments.base import Experiment


class DisplayingSwissRollExperiment(Experiment):
    title = '5.0. Displaying Swiss-roll'
    plotting = True

    samples = 10000

    def _run(self):
        data, target = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)

        self.displayer.load(data, target).show()

        print('Correlation matrix:')
        print(np.cov(data, rowvar=0))


if __name__ == '__main__':
    DisplayingSwissRollExperiment().start()
