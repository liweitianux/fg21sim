# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license
#
# References:
# [1] K. M. Gorski, et al. 2005, ApJ, 622, 759
#     "HEALPix: A Framework for High-resolution Discretization and Fast
#      Analysis of Data Distributed on the Sphere"
#     http://healpix.sourceforge.net/
# [2] M. R. Calabretta & B. F. Roukema 2007, MNRAS, 381, 865
#     "Mapping on the HEALPix Grid"
# [3] M. R. Calabretta: WCSLIB: HPXcvt
#     http://www.atnf.csiro.au/people/mcalabre/WCS/

"""
HEALPix utilities
-----------------

healpix2hpx:
  reorganize the HEALPix data (1D array as FITS table) into 2D FITS image
  in HPX coordinate system

hpx2healpix:
  revert the above reorganization and turn the 2D image in HPX format
  back into HEALPix data as 1D array.
"""


from datetime import datetime, timezone
import logging

import numpy as np
import numba as nb
import healpy as hp
from astropy.io import fits

from .fits import read_fits_healpix


logger = logging.getLogger(__name__)


def healpix2hpx(data, append_history=None, append_comment=None):
    """Reorganize the HEALPix data (1D array as FITS table) into 2D FITS
    image in HPX coordinate system.

    Parameters
    ----------
    data : str or `~astropy.io.fits.BinTableHDU`
        The input HEALPix map to be converted to the HPX image,
        which can be either the filename of the HEALPix FITS file,
        or be a `~astropy.io.fits.BinTableHDU` instance containing
        the HEALPix data as well as its header.
    header : `~astropy.io.fits.Header`, optional
        Header of the HEALPix FITS file
    append_history : list[str]
        Append the provided history to the output FITS header
    append_comment : list[str]
        Append the provided comment to the output FITS header

    Returns
    -------
    hpx_data : 2D `~numpy.ndarray`
        The reorganized HPX image
    hpx_header : `~astropy.io.fits.Header`
        FITS header for the HPX image
    """
    hp_data, hp_header = read_fits_healpix(data)
    dtype = hp_data.dtype
    npix = len(hp_data)
    nside = hp.npix2nside(npix)
    logger.info("Loaded HEALPix data: dtype={0}, Npixel={1}, Nside={2}".format(
        dtype, npix, nside))
    hp_data = np.append(hp_data, np.nan).astype(dtype)
    logger.info("Calculating the HPX indexes ...")
    hpx_idx = _calc_hpx_indexes(nside)
    # Fix indexes of "-1" to set empty pixels with above appended NaN
    hpx_idx[hpx_idx == -1] = len(hp_data) - 1
    hpx_data = hp_data[hpx_idx]
    hpx_header = _make_hpx_header(hp_header,
                                  append_history=append_history,
                                  append_comment=append_comment)
    return (hpx_data.astype(hp_data.dtype), hpx_header)


def hpx2healpix(data, append_history=None, append_comment=None):
    """Revert the reorganization and turn the 2D image in HPX format
    back into HEALPix data as 1D array.

    Parameters
    ----------
    data : str or `~astropy.io.fits.PrimaryHDU`
        The input HPX image to be converted to the HEALPix data,
        which can be either the filename of the HPX FITS image,
        or be a `~astropy.io.fits.PrimaryHDU` instance containing
        the HPX image as well as its header.
    append_history : list[str]
        Append the provided history to the output FITS header
    append_comment : list[str]
        Append the provided comment to the output FITS header

    Returns
    -------
    hp_data : 1D `~numpy.ndarray`
        HEALPix data reorganized from the input HPX image
    hp_header : `~astropy.io.fits.Header`
        FITS header for the HEALPix data
    """
    if isinstance(data, str):
        hpx_hdu = fits.open(data)[0]
        hpx_data, hpx_header = hpx_hdu.data, hpx_hdu.header
        logger.info("Read HPX image from FITS file: %s" % data)
    else:
        hpx_data, hpx_header = data.data, data.header
        logger.info("Read HPX image from PrimaryHDU")
    logger.info("HPX image dtype: {0}".format(hpx_data.dtype))
    logger.info("HPX coordinate system: ({0}, {1})".format(
        hpx_header["CTYPE1"], hpx_header["CTYPE2"]))
    if ((hpx_header["CTYPE1"], hpx_header["CTYPE2"]) !=
            ("GLON-HPX", "GLAT-HPX")):
        raise ValueError("only Galactic 'HPX' projection currently supported")
    # Calculate Nside
    nside = round(hpx_header["NAXIS1"] / 5)
    nside2 = round(90 / np.sqrt(2) / hpx_header["CDELT2"])
    if nside != nside2:
        raise ValueError("Cannot determine the Nside value")
    logger.info("Determined HEALPix Nside=%d" % nside)
    #
    npix = hp.nside2npix(nside)
    logger.info("Calculating the HPX indexes ...")
    hpx_idx = _calc_hpx_indexes(nside).flatten()
    hpx_idx_uniq, idxx = np.unique(hpx_idx, return_index=True)
    if np.sum(hpx_idx_uniq >= 0) != npix:
        raise ValueError("Number of pixels does not match indexes")
    hpx_data = hpx_data.flatten()
    hp_data = hpx_data[idxx[hpx_idx_uniq >= 0]]
    hp_header = _make_healpix_header(hpx_header, nside=nside,
                                     append_history=append_history,
                                     append_comment=append_comment)
    return (hp_data.astype(hpx_data.dtype), hp_header)


@nb.jit(nb.int64[:](nb.int64, nb.int64, nb.int64), nopython=True)
def _calc_hpx_row_idx(nside, facet, jmap):
    """Calculate the HEALPix indexes for one row of a facet.

    NOTE
    ----
    * Only RING ordering is currently supported.
    * This function calculates the double-pixelization index then converts
      it to the regular RING index.

    References: ref.[2], Sec.3.1
    """
    I0 = [1,  3, -3, -1,  0,  2,  4, -2,  1,  3, -3, -1]
    J0 = [1,  1,  1,  1,  0,  0,  0,  0, -1, -1, -1, -1]
    #
    n2side = 2 * nside
    n8side = 8 * nside
    nside1 = nside - 1
    # double-pixelization index of the last pixel in the north polar cap
    npole = (n2side - 1) ** 2 - 1
    # double-pixelization pixel coordinates of the center of the facet
    i0 = nside * I0[facet]
    j0 = nside * J0[facet]
    #
    row_idx = np.zeros(nside, dtype=np.int64)
    for imap in range(nside):
        # (i, j) are 0-based, double-pixelization pixel coordinates.
        # The origin is at the intersection of the equator and prime
        # meridian, `i` increases to the east (N.B.) and `j` to the north.
        i = i0 + nside1 - (jmap + imap)
        j = j0 + jmap - imap
        # convert `i` for counting pixels
        if i < 0:
            i += n8side
        i += 1
        #
        if j > nside:
            # north polar regime
            if j == n2side:
                idx2 = 0
            else:
                # number of pixels in a polar facet with this value of `j`
                npj = 2 * (n2side - j)
                # index of the last pixel in the row above this
                idx2 = (npj - 1) ** 2 - 1
                # number of pixels in this row in the polar facets before this
                idx2 += npj * (i // n2side)
                # pixel number in this polar facet
                idx2 += i % n2side - (j - nside) - 1
        elif j >= -nside:
            # equatorial regime
            idx2 = npole + n8side * (nside - j) + i
        else:
            # south polar regime
            idx2 = 24 * nside**2 + 1
            if j > -n2side:
                # number of pixels in a polar facet with this value of `j`
                npj = 2 * (n2side + j)
                # total number of pixels in this row or below it
                idx2 -= (npj + 1) ** 2
                # number of pixels in this row in the polar facets before this
                idx2 += npj * (i // n2side)
                # pixel number in this polar facet
                idx2 += i % n2side + (j + nside) - 1
        # convert double-pixelization index to regular RING index
        idx = (idx2 - 1) // 2
        row_idx[imap] = idx
    return row_idx


@nb.jit(nb.int64[:, :](nb.int64), nopython=True)
def _calc_hpx_indexes(nside):
    """Calculate HEALPix element indexes for the HPX projection scheme.

    Parameters
    ----------
    nside : int
        Nside of the input/output HEALPix data

    Returns
    -------
    indexes : 2D `~numpy.ndarray`
        2D integer array of same size as the input/output HPX FITS image,
        with elements tracking the indexes of the HPX pixels in the
        HEALPix 1D array, while elements with value "-1" indicating
        null/empty HPX pixels.

    NOTE
    ----
    * The indexes are 0-based;
    * Currently only HEALPix RING ordering supported;
    * The null/empty elements in the HPX projection are filled with "-1".
    """
    # number of horizontal/vertical facet
    nfacet = 5
    # Facets layout of the HPX projection scheme.
    # Note that this appears to be upside-down, and the blank facets
    # are marked with "-1".
    # Ref: ref.[2], Fig.4
    #
    # XXX:
    # Cannot use the nested list here, which fails with ``numba`` error:
    # ``NotImplementedError: unsupported nested memory-managed object``
    FACETS_LAYOUT = np.zeros((nfacet, nfacet), dtype=np.int64)
    FACETS_LAYOUT[0, :] = [6,   9, -1, -1, -1]
    FACETS_LAYOUT[1, :] = [1,   5,  8, -1, -1]
    FACETS_LAYOUT[2, :] = [-1,  0,  4, 11, -1]
    FACETS_LAYOUT[3, :] = [-1, -1,  3,  7, 10]
    FACETS_LAYOUT[4, :] = [-1, -1, -1,  2,  6]
    #
    shape = (nfacet*nside, nfacet*nside)
    indexes = -np.ones(shape, dtype=np.int64)
    #
    # Loop vertically facet-by-facet
    for jfacet in range(nfacet):
        # Loop row-by-row
        for j in range(nside):
            row = jfacet * nside + j
            # Loop horizontally facet-by-facet
            for ifacet in range(nfacet):
                facet = FACETS_LAYOUT[jfacet, ifacet]
                if facet < 0:
                    # blank facet
                    pass
                else:
                    idx = _calc_hpx_row_idx(nside, facet, j)
                    col = ifacet * nside
                    indexes[row, col:(col+nside)] = idx
    #
    return indexes


def _make_hpx_header(header, append_history=None, append_comment=None):
    """Make the FITS header for the HPX image."""
    header = header.copy(strip=True)
    nside = header["NSIDE"]
    # set pixel transformation parameters
    crpix1 = (5 * nside + 1) / 2.0
    crpix2 = crpix1
    header["CRPIX1"] = (crpix1, "Coordinate reference pixel")
    header["CRPIX2"] = (crpix2, "Coordinate reference pixel")
    cos45 = np.cos(np.deg2rad(45))
    header["PC1_1"] = (cos45,  "Transformation matrix element")
    header["PC1_2"] = (cos45,  "Transformation matrix element")
    header["PC2_1"] = (-cos45, "Transformation matrix element")
    header["PC2_2"] = (cos45,  "Transformation matrix element")
    cdelt1 = -90.0 / nside / np.sqrt(2)
    cdelt2 = -cdelt1
    header["CDELT1"] = (cdelt1,  "[deg] Coordinate increment")
    header["CDELT2"] = (cdelt2,  "[deg] Coordinate increment")
    # set celestial transformation parameters
    header["CTYPE1"] = ("GLON-HPX",
                        "Galactic longitude in an HPX projection")
    header["CTYPE2"] = ("GLAT-HPX",
                        "Galactic latitude in an HPX projection")
    header["CRVAL1"] = (0.0,
                        "[deg] Galactic longitude at the reference point")
    header["CRVAL2"] = (0.0,
                        "[deg] Galactic latitude at the reference point")
    header["PV2_1"] = (4, "HPX H parameter (longitude)")
    header["PV2_2"] = (3, "HPX K parameter (latitude)")
    logger.info("Made HPX FITS header")
    #
    header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                      "File creation date")
    comments = [
        'The HPX coordinate system is an reorganization of the HEALPix',
        'data without regridding or interpolation, which is described in',
        '"Mapping on the HEALPix Grid" by M. Calabretta and B. Roukema',
        '(2007, MNRAS, 381, 865-872)',
        'See also http://www.atnf.csiro.au/people/Mark.Calabretta/'
    ]
    for comment in comments:
        header.add_comment(comment)
    #
    if append_history is not None:
        logger.info("HPX FITS header: append history")
        for history in append_history:
            header.add_history(history)
    if append_comment is not None:
        logger.info("HPX FITS header: append comments")
        for comment in append_comment:
            header.add_comment(comment)
    return header


def _make_healpix_header(header, nside,
                         append_history=None, append_comment=None):
    """Make the FITS header for the HEALPix data."""
    header = header.copy(strip=True)
    # set HEALPix parameters
    header["PIXTYPE"] = ("HEALPIX", "HEALPix pixelization")
    header["ORDERING"] = ("RING",
                          "Pixel ordering scheme, either RING or NESTED")
    header["NSIDE"] = (nside, "HEALPix resolution parameter")
    npix = hp.nside2npix(nside)
    header["NPIX"] = (npix, "Total number of pixels")
    header["FIRSTPIX"] = (0, "First pixel # (0 based)")
    header["LASTPIX"] = (npix-1, "Last pixel # (0 based)")
    logger.info("Made HEALPix FITS header")
    #
    header["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                      "File creation date")
    #
    if append_history is not None:
        logger.info("HEALPix FITS header: append history")
        for history in append_history:
            header.add_history(history)
    if append_comment is not None:
        logger.info("HEALPix FITS header: append comments")
        for comment in append_comment:
            header.add_comment(comment)
    return header


@nb.jit(nb.int64(nb.int64), nopython=True)
def nside2npix(nside):
    """Calculate the number of pixels for the given Nside resolution.

    NOTE
    ----
    This is the JIT-optimized version that replaces the ``healpy.nside2npix``
    """
    return 12 * nside * nside


@nb.jit(nb.int64(nb.int64, nb.float64, nb.float64), nopython=True)
def ang2pix_ring_single(nside, theta, phi):
    """Calculate the pixel indexes in RING ordering scheme for one single
    pair of angular coordinate on the sphere.

    Parameters
    ----------
    theta : float
        The polar angle (i.e., latitude), θ ∈ [0, π]. (unit: rad)
    phi : float
        The azimuthal angle (i.e., longitude), φ ∈ [0, 2π). (unit: rad)

    Returns
    -------
    ipix : int
        The index of the pixel corresponding to the input coordinate.

    NOTE
    ----
    * Only support the *RING* ordering scheme
    * This is the JIT-optimized version that partially replaces the
      ``healpy.ang2pix``

    References
    ----------
    - HEALPix software: ``src/C/subs/chealpix.c``: ``ang2pix_ring_z_phi()``
      http://healpix.sourceforge.net/
    """
    z = np.cos(theta)  # colatitude
    za = np.absolute(z)
    tt = (2.0 / np.pi) * np.remainder(phi, 2*np.pi)  # range: [0, 4)
    if za <= 2.0/3.0:
        # Equatorial region
        temp1 = nside * (tt + 0.5)
        temp2 = nside * z * 0.75
        jp = int(temp1 - temp2)  # Index of ascending edge line
        jm = int(temp1 + temp2)  # Index of descending edge line
        # Ring number counted from z=2/3
        iring = nside + 1 + jp - jm  # range: [1, 2n+1]
        kshift = 1 - (iring & 1)  # kshift=1 if ir even, 0 otherwise
        ip = int((jp + jm - nside + kshift + 1) / 2)
        ip = np.remainder(ip, 4*nside)
        ipix = nside * (nside-1) * 2 + (iring-1) * 4 * nside + ip
    else:
        # North & South polar caps
        tp = tt - int(tt)
        tmp = nside * np.sqrt(3 * (1-za))
        jp = int(tp * tmp)
        jm = int((1.0-tp) * tmp)
        # Ring number counted from the closest pole
        iring = jp + jm + 1
        ip = int(tt * iring)
        ip = np.remainder(ip, 4*iring)
        #
        if z > 0:
            ipix = 2 * iring * (iring-1) + ip
        else:
            ipix = 12 * nside * nside - 2 * iring * (iring+1) + ip
    #
    return ipix


@nb.jit(nb.types.UniTuple(nb.float64, 2)(nb.int64, nb.int64), nopython=True)
def pix2ang_ring_single(nside, ipix):
    """Calculate the angular coordinate on the sphere for one pixel index
    in the RING ordering scheme.

    Parameters
    ----------
    ipix : int
        The index of the HEALPix pixel in RING ordering.

    Returns
    -------
    theta : float
        The polar angle (i.e., latitude), θ ∈ [0, π]. (unit: rad)
    phi : float
        The azimuthal angle (i.e., longitude), φ ∈ [0, 2π). (unit: rad)

    NOTE
    ----
    * Only support the *RING* ordering scheme
    * This is the JIT-optimized version that partially replaces the
      ``healpy.ang2pix``

    References
    ----------
    - HEALPix software: ``src/C/subs/chealpix.c``: ``pix2ang_ring_z_phi()``
      http://healpix.sourceforge.net/
    """
    ncap = nside * (nside-1) * 2
    npix = nside2npix(nside)
    fact2 = 4.0 / npix
    if ipix < ncap:
        # North polar cap
        tmp = int(np.sqrt(2*ipix + 1 + 0.5))
        # Ring number counted from the North pole
        iring = int((tmp + 1) / 2)
        iphi = (ipix + 1) - 2 * iring * (iring-1)
        z = 1.0 - iring * iring * fact2
        phi = (iphi - 0.5) * np.pi / (2 * iring)
    elif ipix < (npix - ncap):
        # Equatorial region
        fact1 = 2 * nside * fact2
        ip = ipix - ncap
        # Ring number counted from the North pole
        iring = int(ip / (4*nside) + nside)
        iphi = ip % (4*nside) + 1
        if (iring + nside) % 2 == 1:
            fodd = 1.0  # (iring+nside) is odd
        else:
            fodd = 0.5
        z = (2*nside - iring) * fact1
        phi = (iphi - fodd) * np.pi / (2 * nside)
    else:
        # South polar cap
        ip = npix - ipix
        tmp = int(np.sqrt(2*ip - 1 + 0.5))
        # Ring number counted from the South pole
        iring = int((tmp + 1) / 2)
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring-1))
        z = iring * iring * fact2 - 1.0
        phi = (iphi - 0.5) * np.pi / (2 * iring)
    #
    theta = np.arccos(z)
    return (theta, phi)


@nb.jit([nb.int64[:](nb.int64, nb.float64[:], nb.float64[:]),
         nb.int64[:, :](nb.int64, nb.float64[:, :], nb.float64[:, :])],
        nopython=True)
def ang2pix_ring(nside, theta, phi):
    """Calculate the pixel indexes in RING ordering scheme for each
    pair of angular coordinates on the sphere.

    Parameters
    ----------
    theta : 1D or 2D `~numpy.ndarray`
        The polar angles (i.e., latitudes), θ ∈ [0, π]. (unit: rad)
    phi : 1D or 2D `~numpy.ndarray`
        The azimuthal angles (i.e., longitudes), φ ∈ [0, 2π). (unit: rad)

    Returns
    -------
    ipix : 1D or 1D `~numpy.ndarray`
        The indexes of the pixels corresponding to the input coordinates.
        The shape is the same as the input array.

    NOTE
    ----
    * Only support the *RING* ordering scheme
    * This is the JIT-optimized version that partially replaces the
      ``healpy.ang2pix``
    """
    shape = theta.shape
    size = theta.size
    theta = theta.flatten()
    phi = phi.flatten()
    ipix = np.zeros(size, dtype=np.int64)
    for i in range(size):
        ipix[i] = ang2pix_ring_single(nside, theta[i], phi[i])
    return ipix.reshape(shape)


@nb.jit([nb.types.UniTuple(nb.float64[:], 2)(nb.int64, nb.int64[:]),
         nb.types.UniTuple(nb.float64[:, :], 2)(nb.int64, nb.int64[:, :])],
        nopython=True)
def pix2ang_ring(nside, ipix):
    """Calculate the angular coordinates on the sphere for each pixel
    index in the RING ordering scheme.

    Parameters
    ----------
    ipix : 1D or 2D `~numpy.ndarray`
        The indexes of the HEALPix pixels in the RING ordering

    Returns
    -------
    theta : 1D or 2D `~numpy.ndarray`
        The polar angles (i.e., latitudes), θ ∈ [0, π]. (unit: rad)
    phi : 1D or 2D `~numpy.ndarray`
        The azimuthal angles (i.e., longitudes), φ ∈ [0, 2π). (unit: rad)
        The shape is the same as the input array.

    NOTE
    ----
    * Only support the *RING* ordering scheme
    * This is the JIT-optimized version that partially replaces the
      ``healpy.pix2ang``
    """
    shape = ipix.shape
    size = ipix.size
    ipix = ipix.flatten()
    theta = np.zeros(size, dtype=np.float64)
    phi = np.zeros(size, dtype=np.float64)
    for i in range(size):
        theta_, phi_ = pix2ang_ring_single(nside, ipix[i])
        theta[i] = theta_
        phi[i] = phi_
    return (theta.reshape(shape), phi.reshape(shape))
