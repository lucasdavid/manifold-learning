from sklearn import datasets, svm

from experiments.base import ReductionExperiment, LearningExperiment


class SwissRollPCAExperiment(ReductionExperiment, LearningExperiment):
    title = '5.0. Reducing Swiss-roll with PCA Experiment'
    plotting = True

    samples = 1000
    reduction_method = 'pca'

    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _run(self):
        self.load_data()
        self.learn()

        for dimensions in (3, 2, 1):
            self.reduction_params = {'n_components': dimensions}
            self.reduce()
            self.learn()

        self.displayer.show()

    def load_data(self):
        self.data, self.target = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)
        self.original_data = self.data

        self.displayer.load(self.data, self.target, title='Swiss-roll')

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))


if __name__ == '__main__':
    SwissRollPCAExperiment().start()
