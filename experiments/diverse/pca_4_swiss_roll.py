from sklearn import datasets, svm, neighbors

from experiments.base import ReductionExperiment, LearningExperiment


class SwissRollPCAExperiment(ReductionExperiment, LearningExperiment):
    title = 'pca-swiss-roll'
    plotting = True
    reduction_method = 'pca'
    knn = neighbors.KNeighborsRegressor(n_neighbors=1, n_jobs=-1)

    learner = svm.SVR
    learning_parameters = [
        {'C': (1, 10, 100), 'kernel': ('linear',)},
        {'C': (1, 10, 100), 'gamma': (.01, .1), 'kernel': ('rbf',)}
    ]

    def _load_data(self):
        self.data, self.target = datasets.make_swiss_roll(1000, random_state=0)


if __name__ == '__main__':
    SwissRollPCAExperiment().start()
