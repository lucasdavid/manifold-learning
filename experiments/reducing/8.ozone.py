import numpy as np
from sklearn.preprocessing import Imputer
from experiments.base import ReductionExample


class OzoneExample(ReductionExample):
    title = '7. Ozone Example'
    plotting = True

    data_file = '../../datasets/ozone/onehr.data'
    original_data = None

    def read_data(self):
        self.data = np.genfromtxt(self.data_file, delimiter=',', missing_values='?', usecols=range(1, 74))

        self.data = Imputer().fit_transform(self.data)

        self.target = self.data[:, -1]
        self.data = self.original_data = self.data[:, :-1]

        print('Approximately %.2f%% of the %i samples are ozone days.' % (
            self.target.sum() / len(self.target), self.data.shape[0]))

    def _run(self):
        self.read_data()

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        for dimensions in (3, 2,):
            self.reduction_params['n_components'] = dimensions
            self.data = self.original_data
            self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    OzoneExample().start()
