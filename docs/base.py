import abc
import time
import multiprocessing

import numpy as np

import matplotlib.pyplot as plt
from sklearn import grid_search, cross_validation, decomposition
from sklearn.metrics import confusion_matrix

from manifold.infrastructure import Displayer
from manifold.learning.algorithms import Isomap


class Example(metaclass=abc.ABCMeta):
    title = None

    _displayer = None

    @property
    def displayer(self):
        self._displayer = self._displayer or Displayer(t=self.title)
        return self._displayer

    def run(self):
        raise NotImplementedError

    def dispose(self):
        pass

    def start(self):
        print(self.title)

        self.run()
        self.dispose()


class LearningExample(Example, metaclass=abc.ABCMeta):
    learner = data = target = None

    def learn(self):
        start = time.time()
        print('GridSearch started at %s...' % start)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(self.data, self.target, test_size=.2)

        parameters = {
            'C': (1, 10, 100, 1000),
            'gamma': (.001, .01, .1, 1, 10),
            'kernel': ('linear', 'rbf', 'sigmoid'),
        }

        grid = grid_search.GridSearchCV(self.learner(), parameters, n_jobs=multiprocessing.cpu_count())
        grid.fit(X_train, y_train)

        print('\tAccuracy: %.2f%%\n'
              '\tTime elapsed: %.0fs\n'
              '\tBest parameters: %s'
              % (grid.best_score_, time.time() - start, grid.best_params_))

        y_predicted = grid.predict(X_test)

        self.displayer.confusion_matrix(y_test, y_predicted)


class ReductionExample(Example, metaclass=abc.ABCMeta):
    data = reduced_data = target = None

    reduction_method = 'isomap'
    reduction_method_params = {
        'k': 4,
        'to_dimension': 3
    }

    def reduce(self):
        print('Dimensionality reduction process has started... ', end=' ')
        start = time.time()

        if self.reduction_method == 'pca':
            self.reduced_data = decomposition.PCA(**self.reduction_method_params).fit_transform(self.data)
        else:
            self.reduced_data = Isomap(self.data, **self.reduction_method_params).run()

        elapsed = time.time() - start
        print('Done (%f).' % elapsed)

        self.displayer.load(self.reduced_data, self.target, title='Data set reduced by %s' % self.reduction_method)
