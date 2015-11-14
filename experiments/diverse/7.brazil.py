import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from experiments.base import ReductionExample


class BrazilExample(ReductionExample):
    title = '7. Brazil Example'
    data_file = '../../datasets/brazil/bra_Country_en_csv_v2.csv'

    headers, labels = None, None

    target_code = 'NY.GDP.PCAP.CD'
    reduction_method = 'pca'
    reduction_params = {'n_components': 3}
    plotting = True

    def read_data(self):
        self.labels = np.genfromtxt(self.data_file, dtype=str, delimiter='","', skip_header=5, usecols=range(2, 4))
        self.data = np.genfromtxt(self.data_file, dtype=float, delimiter='","', skip_header=5, usecols=range(4, 60)) \
            .transpose()

        target_column = np.where(self.labels == self.target_code)[0][0]
        print(target_column)
        self.target = self.data[:, target_column]
        self.data = np.delete(self.data, target_column, axis=1)

        self.data = Imputer(copy=False).fit_transform(self.data)
        print('Data size: (%s), %.2f KB.' % (self.data.shape, (self.data.nbytes / 1024)))

        print(self.target_code)
        print(self.target)

    def plot_target(self):
        plt.subplot(111)
        plt.plot(self.target, lw=4, color='orange')
        plt.show()

    def _run(self):
        self.read_data()
        # self.plot_target()

        self.reduce()

        if self.plotting:
            self.displayer.aspect = (20, 30)
            self.displayer.render()


if __name__ == '__main__':
    BrazilExample().start()
