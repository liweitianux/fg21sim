# Copyright (c) 2017,2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Utilities to help analyze the simulation results.
"""

import logging

import numpy as np
from scipy import optimize


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


def loglinfit(x, y,
              xlim=(None, None), ylim=(None, None),
              coef0=(1, 1),
              **kwargs):
    """
    Fit the data points with a log-linear model: y = a * x^b

    Parameters
    ----------
    x, y : list[float]
        The data points.
    xlim, ylim : float tuple/list of length 2, optional
        The minimum/maximum limit of x/y for the fitting.
        Default: (None, None), i.e., use all the data.
    coef0 : float tuple/list of length 2, optional
        The initial values of the coefficients (a0, b0).
        Default: (1, 1)
    **kwargs :
        Extra parameters passed to ``scipy.optimize.least_squares()``.

    Returns
    -------
    coef : (a, b)
        The fitted coefficients.
    err : (a_err, b_err)
        The uncertainties of the coefficients.
    fun : function
        The function with fitted coefficients to calculate the fitted
        values: fun(x).
    """
    def _f_poly1(x, a, b):
        return a + b * x

    x = np.asarray(x)
    y = np.asarray(y)
    xmin, xmax = xlim
    ymin, ymax = ylim
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    if ymin is None:
        ymin = np.min(y)
    if ymax is None:
        ymax = np.max(y)

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    logx = np.log(x[mask])
    logy = np.log(y[mask])

    args = {
        "method": "trf",
        "loss": "soft_l1",
        "f_scale": np.mean(logy),
    }
    args.update(kwargs)
    p, pcov = optimize.curve_fit(_f_poly1, logx, logy, p0=coef0, **args)

    coef = (np.exp(p[0]), p[1])
    perr = np.sqrt(np.diag(pcov))
    err = (np.exp(perr[0]), perr[1])
    fun = lambda x: np.exp(_f_poly1(np.log(x), *p))

    return coef, err, fun
