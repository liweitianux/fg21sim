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
HEALPix utilities:

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
import healpy as hp
from astropy.io import fits


logger = logging.getLogger(__name__)


def healpix2hpx(data, header=None, append_history=None, append_comment=None):
    """Reorganize the HEALPix data (1D array as FITS table) into 2D FITS
    image in HPX coordinate system.

    Parameters
    ----------
    data : str or BinTableHDU or 1D array
      (1) filename of the HEALPix file;
      (2) BinTableHDU containing the HEALPix data and header
      (3) 1D array containing the HEALPix data
    header : astropy.io.fits header
      header of the HEALPix FITS file;
    append_history : string list
      append the provided history to the output FITS header
    append_comment : string list
      append the provided comment to the output FITS header

    Returns
    -------
    (hpx_data, hpx_header) : (2D numpy array, astropy.io.fits header)
    """
    if isinstance(data, str) or isinstance(data, fits.BinTableHDU):
        hp_data, hp_header = hp.read_map(data, nest=False, h=True,
                                         verbose=False)
        hp_header = fits.header.Header(hp_header)
        logger.info("Read HEALPix data from file or HDU")
    else:
        hp_data, hp_header = np.asarray(data), fits.header.Header(header)
        logger.info("Read HEALPix data from array and header")
    logger.info("HEALPix index ordering: %s" % hp_header["ORDERING"])
    if hp_header["ORDERING"] != "RING":
        raise ValueError("only 'RING' ordering currently supported")
    npix = len(hp_data)
    nside = hp.npix2nside(npix)
    logger.info("HEALPix data: Npix=%d, Nside=%d" % (npix, nside))
    if nside != hp_header["NSIDE"]:
        raise ValueError("HEALPix data Nside does not match the header")
    hp_data = np.concatenate([hp_data, [np.nan]])
    hpx_idx = _calc_hpx_indices(nside)
    # fix indices of "-1" to set empty pixels with above appended "nan"
    hpx_idx[hpx_idx == -1] = len(hp_data) - 1
    hpx_data = hp_data[hpx_idx]
    hpx_header = _make_hpx_header(hp_header,
                                  append_history=append_history,
                                  append_comment=append_comment)
    return (hpx_data, hpx_header)


def hpx2healpix(data, header=None, append_history=None, append_comment=None):
    """Revert the reorganization and turn the 2D image in HPX format
    back into HEALPix data as 1D array.

    Parameters
    ----------
    data : str or PrimaryHDU or 2D array
      (1) filename of the HPX file;
      (2) PrimaryHDU containing the HPX image and header
      (3) 2D array containing the HPX image
    header : astropy.io.fits header
      header of the HPX FITS file;
    append_history : string list
      append the provided history to the output FITS header
    append_comment : string list
      append the provided comment to the output FITS header

    Returns
    -------
    (hp_data, hp_header) : (1D numpy array, astropy.io.fits header)
    """
    if isinstance(data, str):
        hpx_hdu = fits.open(data)[0]
        hpx_data, hpx_header = hpx_hdu.data, hpx_hdu.header
        logger.info("Read HPX image from file")
    elif isinstance(data, fits.PrimaryHDU):
        hpx_data, hpx_header = data.data, data.header
        logger.info("Read HPX image from HDU")
    else:
        hpx_data, hpx_header = np.asarray(data), fits.header.Header(header)
        logger.info("Read HPX image from array and header")
    logger.info("HPX coordinate system: ({0}, {1})".format(
        hpx_header["CTYPE1"], hpx_header["CTYPE2"]))
    if ((hpx_header["CTYPE1"], hpx_header["CTYPE2"]) !=
            ("GLON-HPX", "GLAT-HPX")):
        raise ValueError("only Galactic 'HPX' projection currently supported")
    # calculate Nside
    nside = round(hpx_header["NAXIS1"] / 5)
    nside2 = round(90 / np.sqrt(2) / hpx_header["CDELT2"])
    if nside != nside2:
        raise ValueError("Cannot determine the Nside value")
    logger.info("Determined HEALPix Nside=%d" % nside)
    #
    npix = hp.nside2npix(nside)
    hpx_idx = _calc_hpx_indices(nside).flatten()
    hpx_idx_uniq, idxx = np.unique(hpx_idx, return_index=True)
    if np.sum(hpx_idx_uniq >= 0) != npix:
        raise ValueError("Number of pixels does not match indices")
    hpx_data = hpx_data.flatten()
    hp_data = hpx_data[idxx[hpx_idx_uniq >= 0]]
    hp_header = _make_healpix_header(hpx_header, nside=nside,
                                     append_history=append_history,
                                     append_comment=append_comment)
    return (hp_data, hp_header)


def _calc_hpx_indices(nside):
    """Calculate HEALPix element indices for the HPX projection scheme.

    Parameters
    ----------
    nside : int
      Nside of the input/output HEALPix data

    Returns
    -------
    indices : 2D numpy array (int)
      same size as the input/output HPX FITS image, with elements tracking
      the indices of the HPX pixel in the HEALPix 1D array, and elements
      with value "-1" indicating a null/empty HPX pixel.

    NOTE
    ----
    * The indices are 0-based;
    * Currently only HEALPix RING ordering supported;
    * The null/empty elements in the HPX projection are filled with "-1".
    """
    # number of horizontal/vertical facet
    nfacet = 5
    # Facets layout of the HPX projection scheme.
    # Note that this appears to be upside-down, and the blank facets
    # are marked with "-1".
    # Ref: ref.[2], Fig.4
    FACETS_LAYOUT = [[ 6,  9, -1, -1, -1],
                     [ 1,  5,  8, -1, -1],
                     [-1,  0,  4, 11, -1],
                     [-1, -1,  3,  7, 10],
                     [-1, -1, -1,  2,  6]]
    #
    shape = (nfacet*nside, nfacet*nside)
    indices = -np.ones(shape).astype(np.int)
    logger.info("HPX indices matrix shape: {0}".format(shape))
    #
    # Loop vertically facet-by-facet
    for jfacet in range(nfacet):
        # Loop row-by-row
        for j in range(nside):
            row = jfacet * nside + j
            # Loop horizontally facet-by-facet
            for ifacet in range(nfacet):
                facet = FACETS_LAYOUT[jfacet][ifacet]
                if facet < 0:
                    # blank facet
                    pass
                else:
                    idx = _calc_hpx_row_idx(nside, facet, j)
                    col = ifacet * nside
                    indices[row, col:(col+nside)] = idx
    #
    return indices


def _calc_hpx_row_idx(nside, facet, jmap):
    """Calculate the HEALPix indices for one row of a facet.

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
    row_idx = []
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
        row_idx.append(idx)
    return row_idx


def _make_hpx_header(header, append_history=None, append_comment=None):
    """Make the FITS header for the HPX image.
    """
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
    """Make the FITS header for the HEALPix data.
    """
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
