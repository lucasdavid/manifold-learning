from sklearn import datasets

from experiments.base import ReductionExperiment, LearningExperiment


class IrisExperiment(ReductionExperiment, LearningExperiment):
    title = 'PCA Iris'
    plotting = True

    reduction_method = 'pca'
    reduction_params = {'n_components': 0}

    def _run(self):
        self.load_data()
        self.evaluate()
        self.learn()

        for d in (3, 2, 1):
            self.reduction_params['n_components'] = d
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target
        self.original_data = self.data
        print('Shape: %s' % str(self.data.shape))
        print('Data set size: %.2f KB' % (self.data.nbytes / 1024))

        self.displayer.load(self.data, self.target,
                            axis_labels=iris.feature_names) \
            .save('datasets/pca_iris') \
            .dispose()


if __name__ == '__main__':
    IrisExperiment().start()
