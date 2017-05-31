#!/usr/bin/env python3
#
# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
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
from setuptools.command.test import test as TestCommand

import fg21sim as pkg


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


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
    include_package_data=True,
    # Do NOT installed as a zipped egg, since Tornado requires direct access
    # to the templates and static files.
    zip_safe=False,
    scripts=os.listdir("bin/"),
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "pandas",
        "astropy",
        "healpy",
        "configobj",
        "beautifulsoup4",
        "requests",
        "tornado",
    ],
    dependency_links=[
        "git+https://github.com/astropy/regions.git",
    ],
    tests_require=["pytest"],
    cmdclass={"test": PyTest},
)
