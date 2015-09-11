# Manifold Learning
## Introduction
Initial programming sketches on Manifold Learning for undergrad C.S. major conclusion report.

## Installing
This project requires [numpy](www.numpy.org) and [scipy](www.scipy.org)!

```shell
python setup.py install
```

## Usage
For full-working examples, take a look at the
[docs](https://github.com/lucasdavid/Manifold-Learning/tree/master/docs) folder.

## Testing
```shell
# Running all tests...
nosetests

# Comparing ours with scikit-learn's Isomap implementation.
nosetests --tests tests.real.sintetic_data_sets_test
```
