from sklearn import datasets, svm

from report_assets.base import ReductionExample, LearningExample


class SwissRollIsomapExample(ReductionExample, LearningExample):
    title = '5. Swiss-roll Isomap example'
    plotting = True

    samples = 1000

    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _run(self):
        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll')
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

        self.reduction_method = 'skisomap'

        self.learn()

        for d in (2, 1):
            self.data = swiss_roll
            self.reduction_params = {'n_components': d, 'n_neighbors': 7}
            self.reduce()

            self.data = self.reduced_data
            self.learn()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    SwissRollIsomapExample().start()
