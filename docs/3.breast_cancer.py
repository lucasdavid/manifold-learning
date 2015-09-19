import time
import multiprocessing

from sklearn import grid_search, svm

from manifold.learning import algorithms
from manifold.infrastructure import Retriever, Displayer

TITLE = '3. Breast-cancer'


def main():
    print(TITLE)

    r = Retriever('../datasets/breast-cancer/wdbc.data', delimiter=',')

    # Remove ids, as they are not correlated in any way with the target feature.
    r.split_column(0)

    # Split target from data and retrieve both.
    # Target feature is actually located in the 2nd column, but considering we
    # had the ids removed, it's now in the 1st one.
    data, diagnosis = r.split_target(0).retrieve()
    data = data.astype(float)

    d = Displayer(tite=TITLE).load('Original data-set', data, diagnosis)

    learn(data, diagnosis, d)

    start = time.time()
    reduced_data = algorithms.Isomap(data, k=4).run()
    elapsed = time.time() - start
    d.load('Isomap (%ss)' % (elapsed / 1000), reduced_data, diagnosis)

    learn(reduced_data, diagnosis, d)

    d.render()


def learn(data, target, d):
    start = time.time()
    print('GridSearch started at %s...' % start)

    param_grid = {
        'C': (10, 100, 1000),
        'gamma': (0.01, 0.1, 1),
        'kernel': ('linear',),
    }

    clf = grid_search.GridSearchCV(svm.SVC(), param_grid, n_jobs=multiprocessing.cpu_count())
    clf.fit(data, target)

    print('\tAccuracy: %.2f%%\n'
          '\tTime elapsed: %.0fs\n'
          '\tBest parameters: %s'
          % (clf.best_score_, time.time() - start, clf.best_params_))

    return clf


if __name__ == '__main__':
    main()
