import numpy as np
from sklearn import preprocessing, datasets
from experiments.base import Experiment


class DisplayingIrisExperiment(Experiment):
    title = '3.1.1. Displaying Iris Data Set'
    plotting = True

    def _run(self):
        iris = datasets.load_iris()

        self.displayer \
            .load(iris.data, iris.target, title='Original') \
            .load(preprocessing.normalize(iris.data), iris.target, title='Normalized') \
            .load(preprocessing.scale(iris.data), iris.target, title='Scaled') \
            .load(preprocessing.scale(preprocessing.normalize(iris.data)), iris.target, title='Standardized') \
            .show()

        print('Correlation of Iris')
        print(np.corrcoef(iris.data, rowvar=0))


if __name__ == '__main__':
    DisplayingIrisExperiment().start()
