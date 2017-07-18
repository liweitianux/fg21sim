# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license


"""
Generic drawers (a.k.a. painters) that draw some commonly used shapes.
"""

import logging

import numpy as np
import numba as nb
from scipy import interpolate


logger = logging.getLogger(__name__)


def circle(radius=None, rprofile=None, fill_value=0.0):
    """
    Draw a (filled) circle at the center of the output grid.
    If ``rprofile`` is supplied, then it is used as the radial values
    for the circle.

    Parameters
    ----------
    radius : int
        The radius of the circle to draw
    rprofile : 1D `~numpy.ndarray`, optional
        The radial values for the circle, and ``radius`` will be ignored
        if specified.
        If not provided, then fill the circle with ones.
    fill_value : float, optional
        Value to be filled to the empty pixels, default 0.0

    Returns
    -------
    img : 2D `~numpy.ndarray`
        Image of size ``(2*radius+1, 2*radius+1)`` with the circle drawn
        at the center.

    NOTE
    ----
    Using a rotational formulation to create the 2D window/image from the
    1D window/profile gives more circular contours, than using the
    "outer product."

    Credit
    ------
    [1] MATLAB - creating 2D convolution filters
        https://cn.mathworks.com/matlabcentral/newsreader/view_thread/23588
    """
    if rprofile is not None:
        if radius is not None:
            logger.warning("circle(): Ignored parameter radius.")
        rprofile = np.asarray(rprofile)
        radius = len(rprofile) - 1

    xsize = 2 * radius + 1
    x = np.arange(xsize) - radius
    xg, yg = np.meshgrid(x, x)
    r = np.sqrt(xg**2 + yg**2)
    ridx = (r <= radius)
    img = np.zeros(shape=(xsize, xsize))
    img.fill(fill_value)
    if rprofile is None:
        img[ridx] = 1.0
    else:
        finterp = interpolate.interp1d(x=np.arange(len(rprofile)),
                                       y=rprofile, kind="linear")
        img[ridx] = finterp(r[ridx])
    return img


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
