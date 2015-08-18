# Manifold Learning
## Introduction
Initial programming sketches on Manifold Learning for undergrad C.S. major conclusion report.

## Usage
This project requires [numpy](www.numpy.org) and [scipy](www.scipy.org)!

```shell
# Install dependencies
pip install -r requirements.txt --upgrade

# Install project
python setup.py install
```

## Testing
```shell
# Running all tests...
nosetests

# Running comparison between sklearn's Isomap implementation:
nosetests --tests tests.real.sintetic_data_sets_test
```
