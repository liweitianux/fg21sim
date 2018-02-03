# Copyright (c) 2016-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Grid utilities.
"""


import numpy as np
# import numba as nb
import healpy as hp
from scipy import ndimage

from .draw import ellipse
# from .transform import rotate_center
# from .healpix import ang2pix_ring


# @nb.jit(nopython=True)
def _wrap_longitudes(lon):
    """Wrap the longitudes for values that beyond the valid range [0, 360)"""
    lon[lon < 0] += 360
    lon[lon >= 360] -= 360
    return lon


# @nb.jit(nopython=True)
def _wrap_latitudes(lat):
    """Wrap the latitudes for values that beyond the valid range [-90, 90]"""
    lat[lat < -90] = -lat[lat < -90] - 180
    lat[lat > 90] = -lat[lat > 90] + 180
    return lat


# @nb.jit(nb.types.UniTuple(nb.float64[:, :], 2)(
#     nb.types.UniTuple(nb.float64, 2),
#     nb.types.UniTuple(nb.float64, 2),
#     nb.float64),
#         nopython=True)
def make_coordinate_grid(center, size, resolution):
    """
    Make a rectangular, Cartesian coordinate grid.

    This is the ``numba.jit`` optimized version of ``make_coordinate_grid``.

    Parameters
    ----------
    center : 2-float tuple
        Center coordinate (longitude, latitude) of the grid,
        with longitude [0, 360) degree, latitude [-90, 90] degree.
    size : float, or 2-float tuple
        The sizes (size_lon, size_lat) of the grid along the longitude
        and latitude directions.
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
    size_lon, size_lat = size
    # Half number of pixels (excluding the center)
    hn_lon = int(np.ceil(0.5*size_lon / resolution))
    hn_lat = int(np.ceil(0.5*size_lat / resolution))
    idx_lon = lon0 + np.arange(-hn_lon, hn_lon+1) * resolution
    idx_lat = lat0 + np.arange(-hn_lat, hn_lat+1) * resolution
    # Fix the longitudes and latitudes to be in the valid ranges
    idx_lon = _wrap_longitudes(idx_lon)
    idx_lat = _wrap_latitudes(idx_lat)
    # XXX: ``numba`` currently does not support ``numpy.meshgrid``
    shape = (len(idx_lat), len(idx_lon))
    lon = np.zeros(shape)
    for i in range(shape[0]):
        lon[i, :] = idx_lon
    lat = np.zeros(shape)
    for i in range(shape[1]):
        lat[:, i] = idx_lat
    return (lon, lat)


def make_ellipse(center, radii, rotation):
    """
    Make a square grid map containing the specified rotated ellipse.

    Parameters
    ----------
    center : 2-int tuple
        The row and column indexes of the ellipse center.
    radii : 2-int tuple
        The (major, minor) axes of the filling ellipse, number of pixels.
    rotation : float
        The rotation angle (unit [ degree ]) of the filling ellipse.

    Returns
    -------
    gridmap : 2D float `~numpy.ndarray`
        The array containing the specified ellipse, where the pixels
        corresponding to the ellipse with positive values, while other pixels
        are zeros.
        This array is rotated from the nominal ellipse of value ones,
        therefore the edges of the rotated ellipse is in fraction (0-1),
        which can be regarded as similar to the sub-pixel rendering.
    """
    rmax = max(radii)
    shape = (rmax*2+1, rmax*2+1)
    rr, cc = ellipse(center, radii, shape=shape)
    gridmap = np.zeros(shape)
    # XXX: ``numba`` only support one advanced index
    for ri, ci in zip(rr, cc):
        gridmap[ri, ci] = 1.0
    # Rotate the ellipse about the grid center
    # gridmap = rotate_center(gridmap, angle=rotation, interp=True,
    #                         reshape=False, fill_value=0.0)
    gridmap = ndimage.rotate(gridmap, angle=rotation, reshape=False, order=1)
    return gridmap


def make_grid_ellipse(center, size, resolution, rotation=0.0):
    """
    Make a square coordinate grid just containing the specified
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
    rotation : float
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

    NOTE
    ----
    The generated grid is square, determined by the major axis of the ellipse,
    therefore, we can simply rotate the ellipse without reshaping.
    """
    size_major = max(size)
    size = (size_major, size_major)
    lon, lat = make_coordinate_grid(center, size, resolution)
    shape = lon.shape
    # Fill the ellipse into the grid
    r0, c0 = np.floor(np.array(shape) / 2.0).astype(np.int64)
    radii = np.ceil(0.5*np.array(size)/resolution).astype(np.int64)
    rr, cc = ellipse((r0, c0), (radii[0], radii[1]), shape=shape)
    gridmap = np.zeros(shape)
    # XXX: ``numba`` only support one advanced index
    for ri, ci in zip(rr, cc):
        gridmap[ri, ci] = 1.0
    # Rotate the ellipse about the grid center
    # gridmap = rotate_center(gridmap, angle=rotation, interp=True,
    #                         reshape=False, fill_value=0.0)
    gridmap = ndimage.rotate(gridmap, angle=rotation, reshape=False, order=1)
    return (lon, lat, gridmap)


# @nb.jit(nb.types.Tuple((nb.int64[:], nb.float64[:]))(
#     nb.types.UniTuple(nb.float64[:, :], 3), nb.int64),
#         nopython=True)
def map_grid_to_healpix(grid, nside):
    """
    Map the filled coordinate grid to the HEALPix map (RING ordering).

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
    # XXX: ``numba`` does not support using 2D array as indexes
    lon = grid[0].flatten()
    lat = grid[1].flatten()
    gridmap = grid[2].flatten()
    phi = np.radians(lon)
    theta = np.radians(90.0 - lat)
    # ipix = ang2pix_ring(nside, theta, phi)
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    # Get the corresponding input grid pixels for each HEALPix pixel
    # XXX: ``numba`` currently does not support ``numpy.unique()``
    ipix_perm = ipix.argsort()
    ipix_sorted = ipix[ipix_perm]
    idx_uniq = np.concatenate((np.array([True]),
                               ipix_sorted[1:] != ipix_sorted[:-1]))
    indexes = ipix_sorted[idx_uniq]
    values = np.zeros(indexes.shape)
    for i, idx in enumerate(indexes):
        # XXX: ``numba`` does not support using 2D array as indexes
        values[i] = np.mean(gridmap[ipix == idx])
    return (indexes, values)
