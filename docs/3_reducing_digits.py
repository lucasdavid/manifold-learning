from sklearn import datasets

from docs.base import ReductionExample


class ReducingLinearDataSets(ReductionExample):
    title = '3. Reducing Linear Data Sets'
    method = 'pca'
    params = {
        'n_components': 3
    }

    def run(self):
        digits = datasets.load_digits(n_class=5)

        self.data, self.target = digits.data, digits.target
        self.displayer \
            .load(self.data[:, :3], self.target, title='Digits f[0, 3)') \
            .load(self.data[:, 3:6], self.target, title='Digits f[3, 6)') \
            .load(self.data[:, 6:9], self.target, title='Digits f[6, 9)') \
            .load(self.data[:, 9:12], self.target, title='Digits f[9, 12)')

        self.reduce()

        self.method = 'isomap'
        self.params['k'] = 15
        self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingLinearDataSets().start()
