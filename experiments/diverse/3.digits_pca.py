from sklearn import datasets

from experiments.base import ReductionExample, LearningExample


class DigitsExample(ReductionExample, LearningExample):
    title = '3. Digits PCA Example'

    def _run(self):
        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target
        print('Data set size: %i' % self.data.nbytes)

        self.learn()

        self.reduction_method = 'pca'

        for dimensions in (10, 3, 2, 1):
            self.data = digits.data
            self.reduction_params = {'n_components': dimensions}
            self.reduce()

            self.data = self.reduced_data
            self.learn()

        self.displayer.render()


if __name__ == '__main__':
    DigitsExample().start()
