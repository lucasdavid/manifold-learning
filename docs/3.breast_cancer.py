import time
from manifold.learning import algorithms
from manifold.infrastructure import Retriever, Displayer

TITLE = '3. Breast-cancer'


def main():
    print(TITLE)

    r = Retriever('../datasets/breast-cancer/wdbc.data', target_column=1, delimiter=',')

    # Split target (2nd feature) and remove ids.
    r.split_target()  # .split_column(0)

    data, diagnosis = r.split_target().retrieve()
    data = data.astype(float)

    start = time.time()
    reduced_data = algorithms.Isomap(data, k=10).run()
    elapsed = time.time() - start

    Displayer(tite=TITLE) \
        .load('Original data-set', data, diagnosis) \
        .load('Isomap (%ss)' % (elapsed / 1000), reduced_data, diagnosis) \
        .render()


if __name__ == '__main__':
    main()
