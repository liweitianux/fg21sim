# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Utilities to help analyze the simulation results.
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)


def inverse_cumsum(x):
    """
    Do cumulative sum reversely.

    Credit: https://stackoverflow.com/a/28617608/4856091
    """
    x = np.asarray(x)
    return x[::-1].cumsum()[::-1]


def countdist_integrated(x, nbin, log=True):
    """
    Calculate the integrated counts distribution (i.e., luminosity
    function), representing the counts (number of objects) with a
    greater value.

    Parameters
    ----------
    x : list[float]
        Array of quantities of every object/source.
    nbin : int
        Number of bins to calculate the counts distribution.
    log : bool, optional
        Whether to take logarithm on the ``x`` quantities to determine
        the bin edges?
        Default: True

    Returns
    -------
    counts : 1D `~numpy.ndarray`
        The integrated counts for each bin, of length ``nbin``.
    bins : 1D `~numpy.ndarray`
        The central positions of every bin, of length ``nbin``.
    binedges : 1D `~numpy.ndarray`
        The edge positions of every bin, of length ``nbin+1``.
    """
    x = np.asarray(x)
    if log is True:
        x = np.log(x)

    binedges = np.linspace(x.min(), x.max(), num=nbin+1)
    bins = (binedges[1:] + binedges[:-1]) / 2
    counts, __ = np.histogram(x, bins=binedges)
    # Convert to the integrated counts distribution
    counts = inverse_cumsum(counts)

    if log is True:
        bins = np.exp(bins)
        binedges = np.exp(binedges)

    return (counts, bins, binedges)
