from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from experiments.base import ReductionExperiment, LearningExperiment


class BrazilExperiment(ReductionExperiment, LearningExperiment):
    title = '7. Brazil Example'
    data_file = '../../datasets/brazil/bra_Country_en_csv_v2.csv'

    headers, labels = None, None

    learner = svm.SVR

    target_code = 'NY.GDP.PCAP.CD'
    reduction_method = 'pca'
    reduction_params = {'n_components': 3}
    plotting = True

    def read_data(self):
        self.labels = np.genfromtxt(self.data_file, dtype=str, delimiter='","',
                                    skip_header=5, usecols=range(2, 4))
        data = np.genfromtxt(self.data_file, dtype=float, delimiter='","',
                             skip_header=5, usecols=range(4, 60))

        # Convert to our format.
        data = data.transpose()

        # Remove sample 2015, as it is empty.
        data = data[:-1, :]

        # Extract feature desired.
        target_column = np.where(self.labels == self.target_code)[0][0]

        target = data[:, target_column]
        data = np.delete(data, target_column, axis=1)

        # Imput missing data.
        self.original_data = self.data = Imputer(copy=False).fit_transform(data)
        self.target = Imputer(copy=False).fit_transform(
            target.reshape(-1, 1)).flatten()

        self.displayer \
            .load(self.data, self.target) \
            .save('datasets/brazil') \
            .dispose()

        print('Shape: %s' % str(self.data.shape))
        print('Data size: (%s), %.2f KB.' % (
            self.data.shape, (self.data.nbytes / 1024)))
        print('Target code: %s' % self.target_code)

    def plot_target(self):
        plt.subplot(111)
        plt.plot([1960 + i for i in range(len(self.target))], self.target, lw=8,
                 color='crimson')
        plt.show()

    def _run(self):
        self.read_data()
        self.plot_target()
        self.learn()

        for method, params in (('pca', {}), ('isomap', {'k': 4})):
            self.reduction_method = method
            self.reduction_params = params

            for d in (3, 2, 1):
                params['n_components'] = d
                self.reduce()

        self.displayer.show()


if __name__ == '__main__':
    BrazilExperiment().start()
