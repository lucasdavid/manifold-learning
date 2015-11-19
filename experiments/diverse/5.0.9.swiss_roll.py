from sklearn import datasets, svm

from experiments.base import ReductionExperiment, LearningExperiment


class SwissRollPCAExperiment(ReductionExperiment, LearningExperiment):
    title = '4. Swiss-roll PCA Example'
    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _run(self):
        samples = 1000

        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.original_data = self.data
        self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll')

        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.learn()

        self.reduction_method = 'pca'

        for dimensions in (3, 2, 1):
            self.data = swiss_roll
            self.reduction_params = {'n_components': dimensions}
            self.reduce()
            self.learn()


if __name__ == '__main__':
    SwissRollPCAExperiment().start()
