import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

from report_assets.base import ReductionExample


class ReducingSwissRollExample(ReductionExample):
    title = '2. Reducing Swiss-roll with a linear method'

    def run(self):
        samples = 10000
        plane = np.concatenate(
            (40 * np.random.rand(samples, 1),
             20 * np.random.rand(samples, 1),
             np.zeros((samples, 1))),
            axis=1)

        plane_colors = []
        for point in plane:
            plane_colors.append(np.linalg.norm(point[0]))
        self.data, self.target = plane, plane_colors

        self.displayer.load(plane, plane_colors, title='Plane')

        self.method = 'pca'
        self.params = {'n_components': 1}
        self.reduce()

        swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=samples, random_state=0)
        self.data, self.target = swiss_roll, swiss_roll_colors
        self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll')

        self.reduce()
        self.displayer.render()


if __name__ == '__main__':
    ReducingSwissRollExample().start()
