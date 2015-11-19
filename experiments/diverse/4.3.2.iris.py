from sklearn import datasets

from experiments.base import ReductionExperiment, LearningExperiment


class IrisExperiment(ReductionExperiment, LearningExperiment):
    title = '4.3.2. Iris Flower Data Set'
    plotting = True

    def _run(self):
        self.load_data()

        self.learn()

        self.reduction_method = 'pca'
        self.reduction_params = {'n_components': 2}
        self.reduce()
        self.learn()

        self.reduction_method = 'pca'
        self.reduction_params = {'n_components': 1}
        self.reduce()
        self.learn()

        if self.plotting:
            self.displayer.render()

    def load_data(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target
        self.original_data = self.data
        print('Data set size: %i' % self.data.nbytes)

        if self.plotting:
            self.displayer.load(self.data, self.target)

if __name__ == '__main__':
    IrisExperiment().start()
