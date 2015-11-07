from sklearn import datasets

from report_assets.base import ReductionExample


class NoisySwissRollIsomapExample(ReductionExample):
    title = '7. Noisy Swiss-roll Isomap example'
    plotting = True

    samples = 1000
    noise_range = (.6, 1)
    noise_increment = .2
    neighbors = (7,)

    def _run(self):
        noise = self.noise_range[0]

        while noise <= self.noise_range[1]:
            swiss_roll, swiss_roll_colors = datasets.make_swiss_roll(
                n_samples=self.samples, random_state=0, noise=noise)

            self.data, self.target = swiss_roll, swiss_roll_colors
            self.displayer.load(swiss_roll, swiss_roll_colors, title='Swiss-roll (noise: %.2f)' % noise)
            print('Data set size: %.2fKB' % (self.data.nbytes / 1024))

            self.reduction_method = 'skisomap'

            for n in self.neighbors:
                self.data = swiss_roll
                self.reduction_params = {'n_components': 2, 'n_neighbors': n}
                self.reduce()

            noise += self.noise_increment

            if self.plotting:
                # self.displayer.save(name='7-%.2f.png' % noise).dispose()
                self.displayer.aspect = (10, 70)
                self.displayer.render().dispose()


if __name__ == '__main__':
    NoisySwissRollIsomapExample().start()
