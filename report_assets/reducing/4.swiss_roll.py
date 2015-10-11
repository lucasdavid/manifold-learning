import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from manifold.infrastructure import Displayer

from report_assets.base import ReductionExample


class ReducingSwissRollExample(ReductionExample):
    title = '2. Reducing Swiss-roll with a PCA'

    def _run(self):
        samples = 10000

        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll')

        self.reduction_method = 'pca'

        for dimension in (3, 2, 1):
            self.reduction_params = {'n_components': dimension}
            self.reduce()

        self.displayer.render()


if __name__ == '__main__':
    ReducingSwissRollExample().start()
