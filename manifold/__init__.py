import math
import time
from sklearn import datasets, svm, preprocessing

C = 100.
gamma = .0001
training_percentage = .008


def main():
    iris = datasets.load_digits(n_class=2)

    count = len(iris.data)

    training_samples = math.floor(.5 * count)

    s = svm.SVC()
    s.fit(iris.data[:training_samples], iris.target[:training_samples])

    actual = s.predict(iris.data[training_samples:])
    expected = iris.target[training_samples:]

    print(actual)
    print(expected)

    correctly_classified = [index for index, item in enumerate(actual) if item == expected[index]]
    correctly_count = len(correctly_classified) / len(actual)

    print("%.2f" % correctly_count)


if __name__ == '__main__':
    main()
