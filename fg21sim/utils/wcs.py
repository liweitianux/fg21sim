# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Create WCS for sky projection.
"""

import numpy as np
from astropy.wcs import WCS


def make_wcs(center, size, pixelsize, frame="ICRS", projection="TAN"):
    """
    Make WCS for sky projection usages, etc.

    Parameters
    ----------
    center : (xcenter, ycenter) float tuple
        The equatorial/galactic coordinate of the sky/image center [ deg ].
    size : (xsize, ysize) int tuple
        The size (width, height) of the sky/image.
    pixelsize : float
        The pixel size of the sky/image [ arcmin ]
    frame : str, "ICRS" or "Galactic"
        The coordinate frame, only one of ``ICRS`` or ``Galactic``.
    projection : str, "TAN" or "CAR"
        The projection algorithm used by the sky/image, currently only
        support ``TAN`` (tangential), ``CAR`` (Cartesian).

    Returns
    -------
    w : `~astropy.wcs.WCS`
        Created WCS header/object
    """
    xcenter, ycenter = center  # [ deg ]
    xsize, ysize = size
    delt = pixelsize / 60.0  # [ deg ]
    if projection.upper() not in ["TAN", "CAR"]:
        raise ValueError("unsupported projection: " % projection)
    if frame.upper() == "ICRS":
        ctype = ["RA---" + projection.upper(), "DEC--" + projection.upper()]
    elif frame.upper() == "GALACTIC":
        ctype = ["GLON-" + projection.upper(), "GLAT-" + projection.upper()]
    else:
        raise ValueError("unknown frame: " % frame)

    w = WCS(naxis=2)
    w.wcs.ctype = ctype
    w.wcs.crval = np.array([xcenter, ycenter])
    w.wcs.crpix = np.array([xsize/2.0-0.5, ysize/2.0-0.5])
    w.wcs.cdelt = np.array([-delt, delt])
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.equinox = 2000.0
    return w
