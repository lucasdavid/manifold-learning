import numpy as np
from sklearn import datasets

from experiments.base import ReductionExample


class ReducingSwissRollIsomapExample(ReductionExample):
    title = '5. Reducing The Swiss-roll with Isomap'
    plotting = True
    samples = 1000

    def generate_data(self):
        self.data, self.target = datasets.make_swiss_roll(n_samples=self.samples, random_state=0)
        self.original_data = self.data

        self.displayer.load(self.data, self.target)

        print('Correlation of K')
        print(np.corrcoef(self.data, rowvar=0))

    def _run(self):
        self.generate_data()

        self.reduction_method = 'isomap'

        for d in (2,):
            self.reduction_params = {'n_components': d, 'k': 10}
            self.reduce()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    ReducingSwissRollIsomapExample().start()
