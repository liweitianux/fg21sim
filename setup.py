#!/usr/bin/env python3
#
# References:
# [1] Python Packaging User Guide
#     https://packaging.python.org/
#

import os
import sys

from setuptools import setup, find_packages

import fg21sim as pkg


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Check the minimal Python version
if sys.version_info < (3, 4):
    sys.exit("Sorry, Python >= 3.4 is required...")


setup(
    name=pkg.__pkgname__,
    version=pkg.__version__,
    description=pkg.__description__,
    long_description=read("README.rst"),
    author=pkg.__author__,
    author_email=pkg.__author_email__,
    url=pkg.__url__,
    license=pkg.__license__,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    packages=find_packages(exclude=["docs", "tests"]),
    scripts=[
        "bin/healpix2hpx",
        "bin/hpx2healpix",
    ],
    install_requires=[
        "numpy",
        "astropy",
        "healpy",
    ],
    tests_require=["pytest"],
)
