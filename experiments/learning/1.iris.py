from sklearn import datasets, svm
from sklearn import preprocessing

from experiments.base import LearningExperiment


class LearningIrisExample(LearningExperiment):
    title = '2. Learning the Iris data set'
    learner = svm.SVC
    plotting = True
    test_after_train = True

    def _run(self):
        self.load_data()

        # Executes GridSearch with a fraction of the data set. Then predicts
        # the rest of the samples and constructs the confusion matrix.
        self.learn()

    def load_data(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target

        self.displayer.load(self.data, self.target, title='Glass data-set')
        self.displayer.show().dispose()

    def dispose(self):
        del self.data, self.target


if __name__ == '__main__':
    LearningIrisExample().start()
