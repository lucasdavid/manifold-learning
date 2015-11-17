import abc
import time
import multiprocessing

from scipy.spatial import distance
from sklearn import grid_search, decomposition, svm, manifold
from manifold.infrastructure import Displayer
from manifold.learning.algorithms import Isomap, MDS


class Example(metaclass=abc.ABCMeta):
    title = None

    _displayer = None
    plotting = False

    exporting_path = '../report/img/experiments'

    @property
    def displayer(self):
        self._displayer = self._displayer or Displayer()
        return self._displayer

    def _run(self):
        raise NotImplementedError

    def _dispose(self):
        pass

    def start(self):
        print('%s\n' % self.title)

        self._run()
        self._dispose()


class LearningExample(Example, metaclass=abc.ABCMeta):
    learner = svm.SVC
    data = target = labels = grid = None

    learning_parameters = [
        {'C': (1, 10, 100, 1000), 'kernel': ('linear',)},
        {'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10), 'kernel': ('rbf', 'sigmoid')}
    ]

    def learn(self):
        start = time.time()
        print('GridSearch started at %s...' % start)

        self.grid = grid_search.GridSearchCV(self.learner(), self.learning_parameters,
                                             n_jobs=multiprocessing.cpu_count())
        self.grid.fit(self.data, self.target)

        print('\tAccuracy: %.2f\n'
              '\tTime elapsed: %.2f s\n'
              '\tBest parameters: %s\n'
              % (self.grid.best_score_, time.time() - start, self.grid.best_params_))


class ReductionExample(Example, metaclass=abc.ABCMeta):
    data = original_data = target = None

    reducer = None
    reduction_method = 'isomap'
    reduction_params = {
        'k': 4,
        'n_components': 3
    }

    def reduce(self):
        assert self.reduction_method in ('pca', 'mds', 'isomap', 'skisomap'), 'Unknown reduction method.'

        to_dimension = self.reduction_params['to_dimension'] if 'to_dimension' in self.reduction_params else \
            self.reduction_params['n_components'] if 'n_components' in self.reduction_params else \
                3

        data = self.original_data

        print('Dimensionality reduction process has started.')
        print('\tMethod: %s' % self.reduction_method)
        print('\tR^%i --> R^%i' % (data.shape[1], to_dimension))

        start = time.time()

        if self.reduction_method == 'pca':
            self.reducer = decomposition.PCA(**self.reduction_params)
            self.data = self.reducer.fit_transform(data)

        if self.reduction_method == 'mds':
            self.reducer = MDS(distance.squareform(distance.pdist(self.data)), **self.reduction_params)
            self.data = self.reducer.run()

        elif self.reduction_method == 'skisomap':
            self.reducer = manifold.Isomap(**self.reduction_params)
            self.data = self.reducer.fit_transform(data)

        else:
            self.reducer = Isomap(self.data, **self.reduction_params)
            self.data = self.reducer.run()

        print('\tNew data set\'s size: %.2f KB' % (self.data.nbytes / 1024))
        print('Done (%.2f s).' % (time.time() - start))

        if self.plotting:
            self.displayer.load(self.data, self.target)
