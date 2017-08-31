# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license


"""
Generic drawers (a.k.a. painters) that draw some commonly used shapes.

Credit
------
* scikit-image - skimage/draw/draw.py
  https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py
"""

import logging

import numpy as np
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


def _ellipse_in_shape(shape, center, radii, rotation=0.0):
    """
    Generate coordinates of points within ellipse bounded by shape.

    Parameters
    ----------
    shape : int tuple (nrow, ncol)
        Shape of the input image.  Must be length 2.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in counter-clockwise
        direction, with respect to the column-axis.
        Unit: [deg]

    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within
        the ellipse.

    Credit
    ------
    * scikit-image - skimage/draw/draw.py
    """
    rotation = np.deg2rad(rotation)
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = (((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 +
                 ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2)
    return np.nonzero(distances < 1)


def ellipse(center, radii, shape=None, rotation=0.0):
    """
    Generate coordinates of pixels within the ellipse.

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
    rotation : float, optional
        Set the ellipse rotation in counter-clockwise direction.
        Unit: [deg]

    Returns
    -------
    rr, cc : int `~numpy.ndarray`
        Pixel coordinates of the ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    The ellipse equation::
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other
    hand using these ellipse positions for an image afterwards may lead to
    appearing on the other side of image, because
    ``image[-1, -1] = image[end-1, end-1]``

    Credit
    ------
    * scikit-image - skimage/draw/draw.py
    """
    center = np.asarray(center)
    radii = np.asarray(radii)
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = (abs(radii[0] * np.cos(rotation)) +
                    radii[1] * np.sin(rotation))
    c_radius_rot = (radii[0] * np.sin(rotation) +
                    abs(radii[1] * np.cos(rotation)))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center,
                               radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc
