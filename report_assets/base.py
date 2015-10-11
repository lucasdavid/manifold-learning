import abc
import time
import multiprocessing

from sklearn import grid_search, decomposition, svm, manifold

from manifold.infrastructure import Displayer
from manifold.learning.algorithms import Isomap


class Example(metaclass=abc.ABCMeta):
    title = None

    _displayer = None

    @property
    def displayer(self):
        self._displayer = self._displayer or Displayer()
        return self._displayer

    def _run(self):
        raise NotImplementedError

    def _dispose(self):
        pass

    def start(self):
        print(self.title)

        self._run()
        self._dispose()


class LearningExample(Example, metaclass=abc.ABCMeta):
    learner = svm.SVC
    data = target = None

    learning_parameters = [
        {'C': (1, 10, 100, 1000), 'kernel': ('linear',)},
        {'C': (1, 10, 100, 1000), 'gamma': (.001, .01, .1, 1, 10), 'kernel': ('rbf', 'sigmoid')}
    ]

    def learn(self):
        start = time.time()
        print('GridSearch started at %s...' % start)

        grid = grid_search.GridSearchCV(self.learner(), self.learning_parameters, n_jobs=multiprocessing.cpu_count(),
                                        verbose=-1)
        grid.fit(self.data, self.target)

        print('\tAccuracy: %.2f\n'
              '\tTime elapsed: %.2fs\n'
              '\tBest parameters: %s'
              % (grid.best_score_, time.time() - start, grid.best_params_))


class ReductionExample(Example, metaclass=abc.ABCMeta):
    data = reduced_data = target = None

    reduction_method = 'isomap'
    reduction_params = {
        'k': 4,
        'n_components': 3
    }

    def reduce(self):
        to_dimension = self.reduction_params['to_dimension'] if 'to_dimension' in self.reduction_params else \
            self.reduction_params['n_components'] if 'n_components' in self.reduction_params else \
                3

        print('Dimensionality reduction process has started')
        print('\tMethod: %s' % self.reduction_method)
        print('\tR^%i -to-> R^%i' % (self.data.shape[1], to_dimension))

        start = time.time()

        if self.reduction_method == 'pca':
            self.reduced_data = decomposition.PCA(**self.reduction_params).fit_transform(self.data)
        elif self.reduction_method == 'skisomap':
            self.reduced_data = manifold.Isomap(**self.reduction_params).fit_transform(self.data)
        else:
            self.reduced_data = Isomap(self.data, **self.reduction_params).run()

        elapsed = time.time() - start
        print('\tNew data set\'s size: %iKB' % (self.reduced_data.nbytes / 1024))
        print('Done (%.2fs).' % elapsed)

        self.displayer.load(self.reduced_data, self.target)
