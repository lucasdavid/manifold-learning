# Manifold Learning
## Introduction
Initial programming sketches on Manifold Learning for undergrad C.S. major conclusion report.

## Installing
This project requires [numpy](www.numpy.org) and [scipy](www.scipy.org)!

```shell
python setup.py install
```

## Testing
```shell
# Running all tests...
nosetests

# Running comparison between sklearn's Isomap implementation:
nosetests --tests tests.real.sintetic_data_sets_test
```

## Usage

For full-working examples, take a look at the [docs](https://github.com/lucasdavid/Manifold-Learning/tree/master/docs) folder.

### Retriever
```py
from manifold.infrastructure import Retriever

glass = Retriever('glass.data')

# We remove the first column of the dataset, as it represents 
# the IDs of the samples.
glass.split_column(0)
# Additionally, its last column represents the target feature.
glass.split_target(-1)

X, y = glass.retrieve()

# X contains the feature matrix, whereas y contains 
# the target column removed in the previous step.

```

### Displayer
```py
from manifold.infrastructure import Displayer

# Instantiates a Displayer, 
# loads the glass dataset (only the first 3 axis will be displayed)
# Finally, renders the dataset representation in the R^n (where n in [1, 3]).
d = Displayer(title='A nice test displayer for test purposes') \
    .load(title='Glass dataset', data=X, color=y)
    .render()

# You can as many datasets as you want. That will 
# render more graphs alongside with the first.
X1, y1 = glass.retrieve()
X2, y2 = iris.retrieve()
X3, y3 = brainwaves.retrieve()

d = Displayer(title='A nice test displayer for test purposes') \
    .load(title='Glass dataset', data=X1, color=y1) \
    .load(title='Iris dataset', data=X2, color=y2) \
    .load(title='Brainwaves dataset', data=X3, color=y3) \
    .render()

```

### ISOMAP
```py
from manifold.learning import algorithms

data, colors = Retriever('glass.data').split_target().retrieve()

reduced_data = algorithms \
	.Isomap(features, k=neighbors, to_dimension=2) \
	.run()

Displayer(title='ISOMAP over glass dataset') \
    .load(title='Original dataset', data=data, color=colors) \
    .load(title='Dataset reduced to 2D', data=reduced_data, color=colors) \
    .render()

```
