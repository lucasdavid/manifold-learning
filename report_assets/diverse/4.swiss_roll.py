from sklearn import datasets, svm

from report_assets.base import ReductionExample, LearningExample


class SwissRollPCAExample(ReductionExample, LearningExample):
    title = '3. Swiss-roll PCA example'

    def _run(self):
        reduce_to_dimensions = [3, 2, 1]

        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=1000, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll')

        print('Data set size: %iKB' % (self.data.nbytes / 1024))

        self.learner = svm.SVR
        self.learning_parameters = [
            {'C': (1, 10, 100), 'kernel': ('linear',)},
            {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
        ]
        self.learn()

        self.method = 'pca'

        for d in reduce_to_dimensions:
            self.data = swiss_roll
            self.params = {'n_components': d}
            self.reduce()

            self.data = self.reduced_data
            self.learning_parameters = [
                {'C': (1, 10), 'kernel': ('linear',)},
                {'C': (1, 10), 'gamma': (.01, .1), 'kernel': ('rbf',)}
            ]
            self.learn()


if __name__ == '__main__':
    SwissRollPCAExample().start()
