from sklearn import datasets, svm, neighbors

from experiments.base import ReductionExperiment, LearningExperiment


class SwissRollPCAExperiment(ReductionExperiment, LearningExperiment):
    title = 'PCA Swiss-roll'
    plotting = True

    samples = 1000
    reduction_method = 'pca'
    knn = neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=-1)

    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _run(self):
        self.load_data()
        self.evaluate()
        self.learn()

        for dimensions in (3, 2, 1):
            self.reduction_params = {'n_components': dimensions}
            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        self.data, self.target = datasets.make_swiss_roll(
            n_samples=self.samples, random_state=0)
        self.original_data = self.data

        self.displayer \
            .load(self.data, self.target) \
            .save('datasets/pca_swiss_roll') \
            .dispose()

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    SwissRollPCAExperiment().start()
