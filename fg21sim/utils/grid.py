# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Grid utilities.
"""


import numpy as np
from scipy import ndimage
import healpy as hp
import numba as nb

from .draw import ellipse


@nb.jit(nopython=True)
def _wrap_longitudes(lon):
    """Wrap the longitudes for values that beyond the valid range [0, 360)"""
    lon[lon < 0] += 360
    lon[lon >= 360] -= 360
    return lon


@nb.jit(nopython=True)
def _wrap_latitudes(lat):
    """Wrap the latitudes for values that beyond the valid range [-90, 90]"""
    lat[lat < -90] = -lat[lat < -90] - 180
    lat[lat > 90] = -lat[lat > 90] + 180
    return lat


def make_coordinate_grid(center, size, resolution):
    """Make a rectangle, Cartesian coordinate grid.

    Parameters
    ----------
    center : 2-float tuple
        Center coordinate (longitude, latitude) of the grid,
        with longitude [0, 360) degree, latitude [-90, 90] degree.
    size : float, or 2-float tuple
        The sizes (size_lon, size_lat) of the grid along the longitude
        and latitude directions.  If only one float specified, then the
        grid is square.
    resolution : float
        The grid resolution, unit [ degree ].

    Returns
    -------
    lon : 2D `~numpy.ndarray`
        The array with elements representing the longitudes of each grid
        pixel.  The array is odd-sized, with the input center locating at
        the exact grid central pixel.
        Also, the longitudes are fixed to be in the valid range [0, 360).
    lat : 2D `~numpy.ndarray`
        The array with elements representing the latitudes of each grid
        pixel.
        Also, the latitudes are fixed to be in the valid range [-90, 90].
    """
    lon0, lat0 = center
    try:
        size_lon, size_lat = size
    except (TypeError, ValueError):
        size_lon = size_lat = size
    # Half number of pixels (excluding the center)
    hn_lon = np.ceil(0.5*size_lon / resolution).astype(np.int)
    hn_lat = np.ceil(0.5*size_lat / resolution).astype(np.int)
    idx_lon = lon0 + np.arange(-hn_lon, hn_lon+1) * resolution
    idx_lat = lat0 + np.arange(-hn_lat, hn_lat+1) * resolution
    # Fix the longitudes and latitudes to be in the valid ranges
    idx_lon = _wrap_longitudes(idx_lon)
    idx_lat = _wrap_latitudes(idx_lat)
    lon, lat = np.meshgrid(idx_lon, idx_lat)
    return (lon, lat)


@nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(
    nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
        nopython=True)
def make_coordinate_grid_fast(lon_c, lat_c, size_lon, size_lat, resolution):
    """Make a rectangle, Cartesian coordinate grid.

    This is the ``numba.jit`` optimized version of ``make_coordinate_grid``.

    Parameters
    ----------
    lon_c, lat_c : float
        The longitude and latitude of the center coordinate,
        with longitude [0, 360) degree, latitude [-90, 90] degree.
    size_lon, size_lat : float
        The sizes of the grid along the longitude and latitude directions.
    resolution : float
        The grid resolution, unit [ degree ].

    Returns
    -------
    lon : 2D `~numpy.ndarray`
        The array with elements representing the longitudes of each grid
        pixel.  The array is odd-sized, with the input center locating at
        the exact grid central pixel.
        Also, the longitudes are fixed to be in the valid range [0, 360).
    lat : 2D `~numpy.ndarray`
        The array with elements representing the latitudes of each grid
        pixel.
        Also, the latitudes are fixed to be in the valid range [-90, 90].
    """
    # Half number of pixels (excluding the center)
    hn_lon = int(np.ceil(0.5*size_lon / resolution))
    hn_lat = int(np.ceil(0.5*size_lat / resolution))
    idx_lon = lon_c + np.arange(-hn_lon, hn_lon+1) * resolution
    idx_lat = lat_c + np.arange(-hn_lat, hn_lat+1) * resolution
    # Fix the longitudes and latitudes to be in the valid ranges
    idx_lon = _wrap_longitudes(idx_lon)
    idx_lat = _wrap_latitudes(idx_lat)
    # ``numpy.meshgrid`` currently not supported by ``numba``
    shape = (len(idx_lat), len(idx_lon))
    lon = np.zeros(shape)
    for i in range(shape[0]):
        lon[i, :] = idx_lon
    lat = np.zeros(shape)
    for i in range(shape[1]):
        lat[:, i] = idx_lat
    return (lon, lat)


def make_grid_ellipse(center, size, resolution, rotation=None):
    """Make a square coordinate grid just containing the specified
    (rotated) ellipse.

    Parameters
    ----------
    center : 2-float tuple
        Center coordinate (longitude, latitude) of the grid,
        with longitude [0, 360) degree, latitude [-90, 90] degree.
    size : 2-float tuple
        The (major, minor) axes of the filling ellipse, unit [ degree ].
    resolution : float
        The grid resolution, unit [ degree ].
    rotation : float, optional
        The rotation angle (unit [ degree ]) of the filling ellipse.

    Returns
    -------
    lon : 2D `~numpy.ndarray`
        The array with elements representing the longitudes of each grid
        pixel.  The array is odd-sized and square, with the input center
        locating at the exact grid central pixel.
        Also, the longitudes are fixed to be in the valid range [0, 360).
    lat : 2D `~numpy.ndarray`
        The array with elements representing the latitudes of each grid
        pixel.
        Also, the latitudes are fixed to be in the valid range [-90, 90].
    gridmap : 2D float `~numpy.ndarray`
        The array containing the specified ellipse, where the pixels
        corresponding to the ellipse with positive values, while other pixels
        are zeros.
        This array is rotated from the nominal ellipse of value ones,
        therefore the edges of the rotated ellipse is in fraction (0-1),
        which can be regarded as similar to the sub-pixel rendering.

    NOTE
    ----
    The generated grid is square, determined by the major axis of the ellipse,
    therefore, we can simply rotate the ellipse without reshaping.
    """
    size_major = max(size)
    lon, lat = make_coordinate_grid_fast(center[0], center[1],
                                         size_major, size_major, resolution)
    shape = lon.shape
    # Fill the ellipse into the grid
    r0, c0 = np.floor(np.array(shape) / 2.0).astype(np.int)
    r_radius, c_radius = np.ceil(0.5*np.array(size)/resolution).astype(np.int)
    rr, cc = ellipse(r0, c0, r_radius, c_radius, shape=shape)
    gridmap = np.zeros(shape)
    gridmap[rr, cc] = 1.0
    if rotation is not None:
        # Rotate the ellipse
        gridmap = ndimage.rotate(gridmap, angle=rotation, order=1,
                                 reshape=False)
    return (lon, lat, gridmap)


def map_grid_to_healpix(grid, nside):
    """Map the filled coordinate grid to the HEALPix map (RING ordering).

    Parameters
    ----------
    grid : 3-element tuple
        A 3-element tuple `(lon, lat, gridmap)` that specifies the coordinate
        grid to be mapped, where `lon` and `lat` are the longitudes and
        latitudes of the grid pixels, and `gridmap` is the image to be
        mapped to the HEALPix map.
    nside : int
        Nside of the output HEALPix map.

    Returns
    -------
    indexes : 1D `~numpy.ndarray`
        The indexes of the effective HEALPix pixels that are mapped from
        the input coordinate grid.  The indexes are in RING ordering.
    values : 1D `~numpy.ndarray`
        The values of each output HEALPix pixel with respect the above
        indexes.

    NOTE
    ----
    Generally, the input coordinate grid has higher resolution than the
    output HEALPix map, so down-sampling is performed by averaging the
    pixels that map to the same HEALPix pixel.
    However, note that the total flux is *NOT PRESERVED* for the mapping
    (or reprojection) procedure.

    XXX/TODO:
    - Implement the flux-preserving algorithm (reference ???)
    """
    lon, lat, gridmap = grid
    phi = np.radians(lon)
    theta = np.radians(90.0 - lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    # Get the corresponding input grid pixels for each HEALPix pixel
    indexes, counts = np.unique(ipix, return_counts=True)
    shape = (len(indexes), max(counts))
    datamap = np.zeros(shape) * np.nan
    # TODO: how to avoid this explicit loop ??
    for i, idx in enumerate(indexes):
        pixels = gridmap[ipix == idx]
        datamap[i, :len(pixels)] = pixels
    values = np.nanmean(datamap, axis=1)
    return (indexes, values)
