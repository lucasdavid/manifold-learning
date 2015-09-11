import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

from manifold.infrastructure import Displayer

TITLE = '2. Manifold Plot'
SAMPLES = 10000


def main():
    print(TITLE)

    plane = np.concatenate(
        (40 * np.random.rand(SAMPLES, 1),
         20 * np.random.rand(SAMPLES, 1),
         np.zeros((SAMPLES, 1))),
        axis=1)

    plane_colors = []
    for point in plane:
        plane_colors.append(np.linalg.norm(point[0]))

    p = PCA(n_components=1)
    reduced_plane = p.fit_transform(plane)

    swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=SAMPLES, random_state=0)

    p = PCA(n_components=1)
    reduced_swiss_roll = p.fit_transform(swiss_roll)

    Displayer(tite=TITLE) \
        .load('Plane', plane, plane_colors) \
        .load('Reduced plane', reduced_plane, plane_colors) \
        .load('Swiss roll', swiss_roll, swiss_roll_colors) \
        .load('Reduced swiss roll', reduced_swiss_roll, plane_colors) \
        .render()


if __name__ == '__main__':
    main()
