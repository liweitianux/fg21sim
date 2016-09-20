#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fg21sim',
    version='0.0.1',
    description='Realistic Foregrounds Simulation for EoR 21cm Signal Detection',
    long_description=readme,
    author='Weitian LI',
    author_email='liweitianux@live.com',
    url='https://github.com/liweitianux/fg21sim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
