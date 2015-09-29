from sklearn import datasets, svm

from report_assets.base import LearningExample


class LearningIrisExample(LearningExample):
    title = '2. Learning the Iris data set'
    learner = svm.SVC

    def run(self):
        iris = datasets.load_iris()

        self.data, self.target = iris.data, iris.target
        self.displayer.load(self.data, self.target, title='Glass data-set').render()

        # Executes GridSearch with a fraction of the data set. Then predicts
        # the rest of the samples and constructs the confusion matrix.
        self.learn()

    def dispose(self):
        del self.data, self.target


if __name__ == '__main__':
    LearningIrisExample().start()
