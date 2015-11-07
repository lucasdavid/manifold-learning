import numpy as np
from sklearn import preprocessing, datasets

from report_assets.base import Example


class DisplayingSwissRollExample(Example):
    title = '3. Displaying The Swiss-roll data set'

    def _run(self):
        n = 10000

        data, target = datasets.make_swiss_roll(n_samples=n, random_state=0)
        data = preprocessing.scale(data)
        target = data.sum(axis=1)

        self.displayer.load(data, target).render()

        print('Covariance of K')
        print(np.cov(data, rowvar=0))


if __name__ == '__main__':
    DisplayingSwissRollExample().start()
