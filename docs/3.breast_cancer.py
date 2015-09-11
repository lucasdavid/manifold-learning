import time

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

    start = time.time()
    reduced_data = algorithms.Isomap(data, k=4).run()
    elapsed = time.time() - start

    Displayer(tite=TITLE) \
        .load('Original data-set', data, diagnosis) \
        .load('Isomap (%ss)' % (elapsed / 1000), reduced_data, diagnosis) \
        .render()


if __name__ == '__main__':
    main()
