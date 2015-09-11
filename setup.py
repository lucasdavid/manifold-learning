from setuptools import setup, find_packages

setup(
    name='manifold',
    version='0.5.3',
    packages=find_packages(include=('manifold', 'manifold.*')),
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=(
        'numpy',
        'scipy',
        'matplotlib',
        'sklearn',
        'networkx',
        'fake-factory',
        'nose',
        'nose-parameterized',
        'coverage',
        'radon',
    ),
    license='MIT',
    test_suite='tests.unit',
    classifiers=[
        'Programming Language :: Python',
        'License :: MIT',
        'Natural language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Topic :: Manifold Learning',
    ],
)
