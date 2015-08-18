from setuptools import setup, find_packages

try:
    with open('requirements.txt') as f:
        requirements = f.readlines()
except IOError:
    requirements = []

setup(
    name='manifold',
    version='0.5.2',
    packages=find_packages(exclude=['tests.*', 'tests']),
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=requirements,
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
