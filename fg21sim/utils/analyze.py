# Copyright (c) 2017,2019 Weitian LI <wt@liwt.net>
# MIT License

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


def countdist(x, nbin, log=True, xmin=None, xmax=None):
    """
    Calculate the counts distribution, i.e., a histogram.

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
        The counts in each bin, of length ``nbin``.
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

    if log is True:
        bins = np.exp(bins)
        binedges = np.exp(binedges)

    return counts, bins, binedges


def countdist_integrated(x, nbin, log=True, xmin=None, xmax=None):
    """
    Calculate the integrated counts distribution (e.g., luminosity
    function, mass function), representing the counts with a greater
    value, e.g., N(>flux), N(>mass).
    """
    counts, bins, binedges = countdist(x=x, nbin=nbin, log=log,
                                       xmin=xmin, xmax=xmax)
    counts = inverse_cumsum(counts)
    return counts, bins, binedges


def logfit(x, y):
    """
    Fit the data points with: y = a * x^b

    Parameters
    ----------
    x, y : list[float]
        The data points.

    Returns
    -------
    coef : (a, b)
        The fitted coefficients.
    fp : function
        The function with fitted coefficients to calculate the fitted
        values: fp(x).
    """
    logx = np.log(x)
    logy = np.log(y)
    fit = np.polyfit(logx, logy, deg=1)
    coef = (np.exp(fit[1]), fit[0])
    fp = lambda x: np.exp(np.polyval(fit, np.log(x)))
    return coef, fp
