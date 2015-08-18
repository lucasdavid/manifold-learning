import os
from manifold.infrastructure import Retriever, Displayer

DATA_SETS_DIR = '../datasets'
DATA_SET = 'glass/glass.data'


def main():
    data_set_file = os.path.join(DATA_SETS_DIR, DATA_SET)

    print('Displaying data set {%s} in the Rn' % data_set_file)

    glass = Retriever(data_set_file, delimiter=',')

    # Glass has the samples' ids in the first column.
    glass.split_column(0)
    # Additionally, its last column represents the target feature.
    glass.split_target()

    data, color = glass.retrieve()

    d = Displayer(title=DATA_SET)

    # Scatter all dimensions (3-by-3), using as many graphs as necessary.
    for begin in range(0, glass.features, 3):
        end = min(glass.features, begin + 3)
        d.load('Dimensions: d e [%i, %i]' % (begin+1, end), data[:, begin:end], color=color)

    d.render()


if __name__ == '__main__':
    main()
