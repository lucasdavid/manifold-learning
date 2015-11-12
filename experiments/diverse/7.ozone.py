import numpy as np
from sklearn.preprocessing import Imputer

from experiments.base import ReductionExample, LearningExample


class OzoneExample(ReductionExample, LearningExample):
    title = '7. Ozone Example'
    plotting = True

    reduction_method = 'skisomap'
    reduction_params = {'n_neighbors': 7}

    data_file = '../../datasets/ozone/onehr.data'

    def read_data(self):
        self.data = np.genfromtxt(self.data_file, delimiter=',', missing_values='?', usecols=range(1, 74))

        self.data = Imputer().fit_transform(self.data)

        self.target = self.data[:, -1]
        print('Approximately %.2f%% of the %i samples are ozone days.' % (
            self.target.sum() / len(self.target), self.data.shape[0]))

    def _run(self):
        self.read_data()

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.learn()

        for dimensions in (60, 40, 30, 20, 10, 5):
            self.reduction_params['n_components'] = dimensions
            self.reduce()

            d = self.data
            self.data = self.reduced_data
            self.learn()

            self.data = d
            del self.reduced_data

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    OzoneExample().start()
