from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingDigitsExample(ReductionExample):
    title = '1. Reducing Digits'

    def run(self):
        digits = datasets.load_digits(n_class=5)

        self.data, self.target = digits.data, digits.target
        self.displayer \
            .load(self.data[:, :3], self.target, title='Digits f[0, 3)') \
            .load(self.data[:, 3:6], self.target, title='Digits f[3, 6)') \
            .load(self.data[:, 6:9], self.target, title='Digits f[6, 9)') \
            .load(self.data[:, 9:12], self.target, title='Digits f[9, 12)')

        # Reduce with PCA
        self.method = 'pca'
        self.params = {'n_components': 3}
        self.reduce()

        # Reduce with Isomap
        self.method = 'isomap'
        self.params['k'] = 15
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingDigitsExample().start()
