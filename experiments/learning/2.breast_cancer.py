from experiments.base import LearningExample
from manifold.infrastructure import Retriever


class LearningBreastCancerExample(LearningExample):
    title = '3. Learning Breast-cancer'
    plotting = True
    
    def _run(self):
        r = Retriever('../../datasets/breast-cancer/wdbc.data', delimiter=',')

        # Remove ids, as they are not correlated in any way with the target feature.
        r.split_column(0)

        # Split target from data and retrieve both.
        # Target feature is actually located in the 2nd column, but considering we
        # had the ids removed, it's now in the 1st one.
        self.data, self.target = r.split_target(0).retrieve()
        self.data = self.data.astype(float)

        self.displayer.load(self.data, self.target, 'Breast-cancer')

        self.learn()

        if self.plotting:
            self.displayer.render()


if __name__ == '__main__':
    LearningBreastCancerExample().start()
