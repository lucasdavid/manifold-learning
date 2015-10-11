from sklearn import datasets

from report_assets.base import ReductionExample, LearningExample


class DigitsExample(ReductionExample, LearningExample):
    title = '3. Digits example'

    def _run(self):
        reduce_to_dimensions = [10, 3, 2, 1]

        digits = datasets.load_digits()

        self.data, self.target = digits.data, digits.target
        print('Data set size: %i' % self.data.nbytes)

        self.learn()

        for d in reduce_to_dimensions:
            self.data = digits.data
            self.reduction_params = {'n_components': d, 'k': 10}
            self.reduce()

            self.data = self.reduced_data
            self.learn()

        self.displayer.render()


if __name__ == '__main__':
    DigitsExample().start()
