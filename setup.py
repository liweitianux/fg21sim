#!/usr/bin/env python3

from setuptools import setup, find_packages

import fg21sim as pkg


setup(
    name=pkg.__pkgname__,
    version=pkg.__version__,
    description=pkg.__description__,
    long_description=open("README.rst").read(),
    author=pkg.__author__,
    author_email=pkg.__author_email__,
    url=pkg.__url__,
    license=pkg.__license__,
    packages=find_packages(exclude=("tests", "docs")),
    scripts=[
        "bin/healpix2hpx",
        "bin/hpx2healpix",
    ],
)
