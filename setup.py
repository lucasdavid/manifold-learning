__author__ = 'Lucas'

from setuptools import setup, find_packages

setup(
    name='nappy',
    version='0.5.2',
    packages=find_packages(exclude=['tests.*', 'tests']),
    long_description=open('README.md').read(),
    include_package_data=True,

    classifiers=[
        'Programming Language :: Python',
        'License :: MIT',
        'Natural language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Topic :: Manifold Learning',
    ],
    install_requires=[
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'manifold = manifold:main'
        ],
    },

    license='MIT',

    extras_require={'tests': ['fake-factory', 'nose', 'nose-parameterized', 'coverage', 'radon']},

    test_suite='tests.unit',
)
