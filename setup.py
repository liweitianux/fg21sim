#!/usr/bin/env python3
#
# Copyright (c) 2016-2018 Weitian LI <wt@liwt.net>
# MIT license
#
# References:
# [1] Python Packaging User Guide
#     https://packaging.python.org/
# [2] pytest - Good Integration Practices
#     http://doc.pytest.org/en/latest/goodpractices.html
# [3] setuptools: Building and Distributing Packages with Setuptools
#     https://setuptools.readthedocs.io/en/latest/setuptools.html
#

import os
import sys

from setuptools import setup, find_packages

import fg21sim as pkg


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if sys.version_info < (3, 4):
    sys.exit("%s requires Python >= 3.4 ..." % pkg.__pkgname__)


setup(
    name=pkg.__pkgname__,
    version=pkg.__version__,
    description=pkg.__description__,
    long_description=read("README.rst"),
    author=pkg.__author__,
    author_email=pkg.__author_email__,
    url=pkg.__url__,
    license=pkg.__license__,
    # Trove classifiers (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    # Do NOT installed as a zipped egg, since Tornado requires direct access
    # to the templates and static files.
    zip_safe=False,
    scripts=[os.path.join("bin", script) for script in os.listdir("bin/")],
    install_requires=[
        "Cython",
        "astropy",
        "beautifulsoup4",
        "configobj",
        "healpy",
        "hmf",
        "numba",
        "numpy",
        "pandas",
        "reproject",
        "requests",
        "scipy",
        "tornado",
    ],
    dependency_links=[
        "git+https://github.com/astropy/regions.git",
    ],
)
