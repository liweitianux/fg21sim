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


def countdist_integrated(x, nbin, log=True, xmin=None, xmax=None):
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
    xmin, xmax : float, optional
        The lower and upper boundaries within which to calculate the
        counts distribution.  They are default to the minimum and
        maximum of the given ``x``.

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
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    x = x[(x >= xmin) & (x <= xmax)]

    if log is True:
        if xmin <= 0:
            raise ValueError("log=True but x have elements <= 0")
        x = np.log(x)
        xmin, xmax = np.log([xmin, xmax])

    binedges = np.linspace(xmin, xmax, num=nbin+1)
    bins = (binedges[1:] + binedges[:-1]) / 2
    counts, __ = np.histogram(x, bins=binedges)
    # Convert to the integrated counts distribution
    counts = inverse_cumsum(counts)

    if log is True:
        bins = np.exp(bins)
        binedges = np.exp(binedges)

    return (counts, bins, binedges)
