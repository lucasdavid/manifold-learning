from sklearn import datasets, neighbors, model_selection
from experiments.base import ReductionExperiment, LearningExperiment


class LeukemiaExperiment(ReductionExperiment, LearningExperiment):
    title = 'Leukemia Isomap'
    plotting = True

    reduction_method = 'isomap'
    reduction_params = {'n_components': 0, 'k': 5}

    knn = neighbors.KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

    learning_parameters = [
        {'kernel': ('linear',), 'C': (1, 10, 100, 1000)},
        {'kernel': ('poly',), 'degree': (2, 3, 4), 'coef0': (0, .1, 1, 10)},
        {'kernel': ('rbf',), 'C': (1, 10, 100),
         'gamma': (.001, .01, .1, 1, 10)},
        {'kernel': ('sigmoid',), 'C': (1, 10, 100, 1000),
         'gamma': (.001, .01, .1, 1, 10), 'coef0': (0, .1, 1, 10)},
    ]

    def _run(self):
        self.load_data()

        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=0)

        print('1-NN score: %.2f'
              % self.knn.fit(X_train, y_train).score(X_test, y_test))

        self.learn()

        for d in (30, 20, 10):
            self.reduction_params['n_components'] = d

            self.reduce()
            self.learn()

        self.displayer.save(self.title)

    def load_data(self):
        leukemia = datasets.fetch_mldata('leukemia', transpose_data=True)
        self.data, self.target = leukemia.data, leukemia.target
        self.original_data = self.data

        self.displayer \
            .load(self.data[:, 1:4], self.target) \
            .save('datasets/leukemia') \
            .dispose()

        print('Shape: %s' % str(self.data.shape))
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))
        print(self.data)
        print(self.target)


if __name__ == '__main__':
    LeukemiaExperiment().start()
