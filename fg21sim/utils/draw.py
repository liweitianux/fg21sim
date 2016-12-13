# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license


"""
Generic drawers (a.k.a. painters) that draw some commonly used shapes.
"""


import numpy as np
import numba as nb


@nb.jit(nb.types.UniTuple(nb.int64[:], 2)(nb.types.UniTuple(nb.int64, 2),
                                          nb.types.UniTuple(nb.int64, 2),
                                          nb.types.UniTuple(nb.int64, 2)),
        nopython=True)
def ellipse(center, radii, shape):
    """
    Generate coordinates of pixels within the ellipse.

    XXX/NOTE
    --------
    * Cannot figure out why ``nb.optional(nb.types.UniTuple(nb.int64, 2))``
      does NOT work.  Therefore, make ``shape`` as mandatory parameter
      instead of optional.

    Parameters
    ----------
    center : int tuple (r0, c0)
        Center coordinate of the ellipse.
    radii : int tuple (r_radius, c_radius)
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 <= 1``.
    shape : int tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  This is useful for ellipses that exceed the image
        size.  If None, the full extent of the ellipse is used.

    Returns
    -------
    rr, cc : int `~numpy.ndarray`
        Pixel coordinates of the ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    """
    # XXX: ``numba`` currently does not support ``numpy.meshgrid``
    nrow, ncol = shape
    r_lim = np.zeros((nrow, ncol))
    for i in range(nrow):
        r_lim[i, :] = np.arange(float(ncol))
    c_lim = np.zeros((nrow, ncol))
    for i in range(ncol):
        c_lim[:, i] = np.arange(float(nrow))

    r_o, c_o = center
    r_r, c_r = radii
    distances = (((r_lim-r_o) / r_r) * ((r_lim-r_o) / r_r) +
                 ((c_lim-c_o) / c_r) * ((c_lim-c_o) / c_r))
    r_idx, c_idx = np.nonzero(distances <= 1.0)
    return (r_idx, c_idx)
