import numpy as np
from sklearn import preprocessing, datasets

from report_assets.base import Example


class DisplayingSManifoldExample(Example):
    title = '4. Displaying the S manifold'

    def _run(self):
        n = 10000

        data, target = datasets.make_s_curve(n_samples=n, noise=0)

        self.displayer.load(data, target).aspect = (10, 60)
        self.displayer.render()

        print('Covariance of K')
        print(np.cov(data, rowvar=0))


if __name__ == '__main__':
    DisplayingSManifoldExample().start()
