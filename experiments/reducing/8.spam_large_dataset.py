from sklearn import datasets
from experiments.base import ReductionExample
from manifold.infrastructure import Retriever


class SpamExample(ReductionExample):
    title = '7. Spam Reduced Example'
    plotting = True
    file = '../datasets/spam/spambase.data'

    def load_data(self):
        self.data, self.target = Retriever(self.file, delimiter=',').split_target().retrieve()

        self.displayer.load(self.data[:, 1:4], self.target)
        print('Data set size: %.2fKB' % (self.data.nbytes / 1024))
        print(self.target)

    def _run(self):
        self.load_data()

        for m, params in (
                ('skisomap', {'n_components': 0, 'n_neighbors': 7}),
                ('isomap', {'n_components': 0, 'k': 7})
        ):
            self.reduction_method = m
            self.reduction_params = params

            self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    SpamExample().start()
