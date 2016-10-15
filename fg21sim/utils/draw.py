# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Generic drawers (i.e., painters) that draw some commonly used shapes.

Credits:
- scikit-image: draw
  http://scikit-image.org/docs/dev/api/skimage.draw.html
  https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
"""


import numpy as np


def _ellipse_in_shape(shape, center, radii):
    """Generate coordinates of points within the ellipse bounded by shape."""
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_o, c_o = center
    r_r, c_r = radii
    distances = ((r_lim - r_o) / r_r)**2 + ((c_lim - c_o) / c_r)**2
    return np.nonzero(distances < 1.0)


def ellipse(r, c, r_radius, c_radius, shape=None):
    """Generate coordinates of pixels within the ellipse.

    Parameters
    ----------
    r, c : float
        Center coordinate of the ellipse.
    r_radius, c_radius : float
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
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
    upper_left = np.ceil(center - radii).astype(int)
    lower_right = np.floor(center + radii).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def circle(r, c, radius, shape=None):
    """Generate coordinates of pixels within the circle.

    Parameters
    ----------
    r, c : float
        Center coordinate of the circle.
    radius : float
        Radius of the circle.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.  This is useful for circles that exceed the image
        size.  If None, the full extent of the circle is used.

    Returns
    -------
    rr, cc : integer `~numpy.ndarray`
        Pixel coordinates of the circle.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from fg21sim.utils.draw import circle
    >>> img = np.zeros((10, 10), dtype=np.uint8)
    >>> rr, cc = circle(4, 4, 5)
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    return ellipse(r, c, radius, radius, shape)
