#!/usr/bin/env python

import io
import re
from glob import glob
from os.path import basename, dirname, join, splitext

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='read_clustering',
    version='0.0.1',
    license='MIT License',
    description='Cluster variant call files and predict modification profiles.',
    author='Andrew Bailey, Shreya Mantripragada, Alejandra Duran, Abhay Padiyar',
    author_email='bailey.andrew4@gmail.com',
    url='https://github.com/adbailey4/read_clustering',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    project_urls={
        'Issue Tracker': 'https://github.com/adbailey4/read_clustering/issues',
    },
    keywords=['variant', 'modification'],
    python_requires='>=3.5',
    install_requires=[
        'pandas>=1.0.5',
        'numpy>=1.14.2',
        'matplotlib>=3.2.2',
        'seaborn>=0.10.1',
        'scikit-learn>=0.23.1, <0.24.0',
        'hdbscan>=0.8.26',
        'yellowbrick>=1.1',
        'scipy>=1.5.0',
        'shapely>=1.7.0',
        'kneed>=0.6.0'],
    setup_requires=['Cython>=0.29.21']
)
