from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from experiments.base import CompleteExperiment


class BrazilExperiment(CompleteExperiment):
    title = 'iso-brazil'
    data_file = '../../datasets/brazil/bra_Country_en_csv_v2.csv'
    plotting = True
    headers, labels = None, None
    learner = svm.SVR

    target_code = 'NY.GDP.PCAP.CD'
    reduction_method = 'isomap'
    reduction_params = {'k': 4}

    def _load_data(self):
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


if __name__ == '__main__':
    BrazilExperiment().start()
