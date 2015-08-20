import numpy as np
from sklearn import datasets
from manifold.infrastructure import Displayer

TITLE = '2. Manifold Plot'
SAMPLES = 10000


def main():
    print(TITLE)

    plane = np.concatenate(
        (20 * np.random.rand(SAMPLES, 1),
         10 * np.random.rand(SAMPLES, 1),
         np.zeros((SAMPLES, 1))),
        axis=1)

    plane_colors = []
    for point in plane:
        plane_colors.append(np.linalg.norm(point[0]))

    swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(n_samples=SAMPLES, random_state=0)
    Displayer(tite=TITLE) \
        .load('swiss_roll-set contained in R^2', plane, plane_colors) \
        .load('Swiss roll with %i samples.' % SAMPLES, swiss_roll, swiss_roll_colors) \
        .render()


if __name__ == '__main__':
    main()
