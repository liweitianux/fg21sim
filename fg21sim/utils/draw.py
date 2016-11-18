# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license


"""
Generic drawers (a.k.a. painters) that draw some commonly used shapes.


Credits
-------
The ``_ellipse_in_shape`` and ``ellipse()`` functions are originally taken
from project [scikit-image]_, which are licensed under the *Modified BSD*
license.

.. [scikit-image] skimage.draw.draw
   http://scikit-image.org/docs/dev/api/skimage.draw.html
   https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
"""


import numpy as np
import numba as nb


@nb.jit([nb.types.UniTuple(nb.int64[:], 2)(nb.types.UniTuple(nb.int64, 2),
                                           nb.types.UniTuple(nb.int64, 2),
                                           nb.types.UniTuple(nb.int64, 2)),
         nb.types.UniTuple(nb.int64[:], 2)(nb.int64[:], nb.int64[:],
                                           nb.int64[:])],
        nopython=True)
def _ellipse_in_shape(shape, center, radii):
    """Generate coordinates of points within the ellipse bounded by shape."""
    # XXX: ``numba`` currently does not support ``numpy.meshgrid``
    nrow, ncol = shape
    r_lim = np.zeros((nrow, ncol))
    for i in range(nrow):
        r_lim[i, :] = np.arange(float(ncol))
    c_lim = np.zeros((nrow, ncol))
    for i in range(ncol):
        c_lim[:, i] = np.arange(float(nrow))
    #
    r_o, c_o = center
    r_r, c_r = radii
    distances = (((r_lim-r_o) / r_r) * ((r_lim-r_o) / r_r) +
                 ((c_lim-c_o) / c_r) * ((c_lim-c_o / c_r)))
    xi, yi = np.nonzero(distances < 1.0)
    return (xi, yi)


@nb.jit(nb.types.UniTuple(nb.int64[:], 2)(nb.int64, nb.int64,
                                          nb.int64, nb.int64,
                                          nb.types.UniTuple(nb.int64, 2)),
        nopython=True)
def ellipse(r, c, r_radius, c_radius, shape):
    """Generate coordinates of pixels within the ellipse.

    XXX/NOTE
    --------
    * Cannot figure out why ``nb.optional(nb.types.UniTuple(nb.int64, 2))``
      does NOT work.  Therefore, make ``shape`` as mandatory parameter
      instead of optional.
    * Cannot figure out multi-dispatch that allows both int and float types
      for ``r``, ``c``, ``r_radius`` and ``c_radius``.  Thus only support
      the int type for the moment.

    Parameters
    ----------
    r, c : int
        Center coordinate of the ellipse.
    r_radius, c_radius : int
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  This is useful for ellipses that exceed the image
        size.  If None, the full extent of the ellipse is used.

    Returns
    -------
    rr, cc : integer `~numpy.ndarray`
        Pixel coordinates of the ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from fg21sim.utils.draw import ellipse
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 5, 3, 4)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])

    # The upper_left and lower_right corners of the
    # smallest rectangle containing the ellipse.
    upper_left = np.ceil(center - radii).astype(np.int64)
    lower_right = np.floor(center + radii).astype(np.int64)

    # Constrain upper_left and lower_right by shape boundary.
    upper_left = np.maximum(upper_left, np.array([0, 0]))
    lower_right = np.minimum(lower_right, np.array(shape)-1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii)
    rr += upper_left[0]
    cc += upper_left[1]
    return (rr, cc)
