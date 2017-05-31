# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
WGS84 Earth geodetic coordinate conversion utilities.

NOTE
----
The WGS84 coordinates (φ, λ, h) are *ellipsoidal*, not spheroidal
(geocentric).

References
----------
[1] Converting GPS Coordinates (φλh) to Navigation Coordinates (ENU)
    http://digext6.defence.gov.au/dspace/bitstream/1947/3538/1/DSTO-TN-0432.pdf
[2] Convert WGS-84 geodetic locations to Cartesian coordinates
    in a local tangent plane
    https://gist.github.com/govert/1b373696c9a27ff4c72a
"""

import numpy as np


class Earth:
    # WGS84 Earth semi-major & semi-minor axis [m]
    a = 6378137.0
    b = 6356752.3142
    # Ellipsoid flatness
    f = (a-b) / a
    # Eccentricity
    e2 = 1.0 - (b/a)**2
    e = e2 ** 0.5


def geodetic2ecef(p):
    """
    Convert the WGS84 geodetic coordinate to the ECEF (Earth Centered
    Earth Fixed) Cartesian coordinate.

    Parameters
    ----------
    p : (lon, lat, h)-tuple
        The WGS84 geodetic point to be converted to ENU coordinate,
        units: lon, lat -> [deg]; h -> [m]

    Returns
    -------
    ecef : (x, y, z)-tuple
        The converted ECEF coordinate. unit: [m]
    """
    lon, lat, h = p
    phi, lam = np.deg2rad([lon, lat])
    sin_phi, sin_lam = np.sin([phi, lam])
    cos_phi, cos_lam = np.cos([phi, lam])
    chi = np.sqrt(1.0 - Earth.e2 * sin_lam * sin_lam)
    v = Earth.a / chi
    x = (v + h) * cos_lam * cos_phi
    y = (v + h) * cos_lam * sin_phi
    z = (v*(1-Earth.e2) + h) * sin_lam
    return (x, y, z)


def geodetic2enu(p, ref):
    """
    Convert the WGS84 geodetic coordinate (longitude, latitude, height)
    to East-North-Up coordinates in a local tangent plane that is
    centered at the reference WGS84 geodetic point.

    Parameters
    ----------
    p : (lon, lat, h)-tuple
        The WGS84 geodetic point to be converted to ENU coordinate,
        units: lon, lat -> [deg]; h -> [m]
    ref : (lon0, lat0, h0)-tuple
        The reference WGS84 geodetic point to determine the local
        tangent plane.

    Returns
    -------
    enu : (east, north, up)-tuple
        The converted ENU coordinate in the determined local tangent
        plane. unit: [m]
    """
    pxyz = np.array(geodetic2ecef(p))
    pxyz0 = np.array(geodetic2ecef(ref))
    dxyz = pxyz - pxyz0

    lon0, lat0, h0 = ref
    phi0, lam0 = np.deg2rad([lon0, lat0])
    sin_phi0, sin_lam0 = np.sin([phi0, lam0])
    cos_phi0, cos_lam0 = np.cos([phi0, lam0])
    # Rotation
    M = np.array([[-sin_phi0,          cos_phi0,          0.0],
                  [-cos_phi0*sin_lam0, -sin_phi0*sin_lam0, cos_lam0],
                  [cos_phi0*cos_lam0,  sin_phi0*cos_lam0,  sin_lam0]])
    east, north, up = M.dot(dxyz)
    return (east, north, up)
