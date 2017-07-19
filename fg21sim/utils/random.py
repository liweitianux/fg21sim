# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Random number and/or points generations.
"""

import numpy as np


def spherical_uniform(n=1):
    """
    Uniformly pick random points on the surface of an unit sphere.
    The algorithm is described in [SpherePointPicking]_.

    Parameters
    ----------
    n : int
        Number of points to be randomly picked

    Returns
    -------
    theta : float, or 1D `~numpy.ndarray`
        The polar angles, θ ∈ [0, π].
        If ``n > 1``, then returns a 1D array containing all the generated
        coordinates.
        Unit: [rad]
    phi : float, or 1D `~numpy.ndarray`
        The azimuthal angles, φ ∈ [0, 2π).
        Unit: [rad]

    NOTE
    ----
    Physicists usually adopt the (radial, polar, azimuthal) order with
    the (r, θ, φ) notation for the spherical coordinates convention, which
    is adopted here and by ``healpy``.
    However, this convention is *different* to the convention generally
    used by mathematicians.

    The following relation can be used to convert the generated (theta, phi)
    to the Galactic/equatorial longitude and latitude convention:
        lon = np.rad2deg(phi)
        lat = 90.0 - np.rad2deg(theta)

    References
    ----------
    .. [SpherePointPicking]
       Wolfram MathWorld - Sphere Point Picking
       http://mathworld.wolfram.com/SpherePointPicking.html

    .. [SphericalCoordinates]
       Wolfram MathWorld - Spherical Coordinates
       http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    u = np.random.uniform(size=n)
    v = np.random.uniform(size=n)
    phi = 2*np.pi * u
    theta = np.arccos(2*v - 1)
    return (theta, phi)
