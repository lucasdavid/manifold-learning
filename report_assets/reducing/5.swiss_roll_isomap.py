import numpy as np
from sklearn import datasets

from report_assets.base import ReductionExample


class ReducingSwissRollIsomapExample(ReductionExample):
    title = '5. Reducing The Swiss-roll with Isomap'

    def _run(self):
        n = 1000

        self.data, self.target = datasets.make_swiss_roll(n_samples=n, random_state=0)
        self.displayer.load(self.data, self.target)

        print('Covariance of K')
        print(np.cov(self.data, rowvar=0))

        self.reduction_method = 'isomap'

        for d in (2, 1):
            self.reduction_params = {'n_components': d, 'k': 7}
            self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingSwissRollIsomapExample().start()
