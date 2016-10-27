# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
FITS utilities
--------------

read_fits_healpix:
    Read the HEALPix map from a FITS file or a BinTableHDU to 1D array
    in *RING* ordering.

write_fits_healpix:
    Write the HEALPix map to a FITS file with proper header as well
    as the user-provided header.
"""

from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
import healpy as hp


# Column formats for FITS binary table
# Reference:
# http://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
FITS_COLUMN_FORMATS = {
    np.dtype("bool"):       "L",
    np.dtype("uint8"):      "B",
    np.dtype("int16"):      "I",
    np.dtype("int32"):      "J",
    np.dtype("int64"):      "K",
    np.dtype("float32"):    "E",
    np.dtype("float64"):    "D",
    np.dtype("complex64"):  "C",
    np.dtype("complex128"): "M",
}


def read_fits_healpix(filename):
    """Read the HEALPix map from a FITS file or a BinTableHDU to 1D array
    in *RING* ordering.

    Parameters
    ----------
    filename : str or `~astropy.io.fits.BinTableHDU`
        Filename of the HEALPix FITS file,
        or an `~astropy.io.fits.BinTableHDU` instance.

    Returns
    -------
    data : 1D `~numpy.ndarray`
        HEALPix data in *RING* ordering with same dtype as input
    header : `~astropy.io.fits.Header`
        Header of the input FITS file

    NOTE
    ----
    This function wraps on `healpy.read_map()`, but set the data type of
    data array to its original value as in FITS file, as well as return
    FITS header as `~astropy.io.fits.Header` instance.
    """
    if isinstance(filename, fits.BinTableHDU):
        hdu = filename
    else:
        # Read the first extended table
        hdu = fits.open(filename)[1]
    # Hack to ignore the dtype byteorder, use native endianness
    dtype = np.dtype(hdu.data.field(0).dtype.type)
    header = hdu.header
    data = hp.read_map(hdu, nest=False, verbose=False)
    return (data.astype(dtype), header)


def write_fits_healpix(filename, hpmap, header=None, clobber=False,
                       checksum=False):
    """Write the HEALPix map to a FITS file with proper header as well
    as the user-provided header.

    This function currently only support one style of HEALPix with the
    following specification:
    - Only one column: I (intensity)
    - ORDERING: RING
    - COORDSYS: G (Galactic)
    - OBJECT: FULLSKY
    - INDXSCHM: IMPLICIT

    Parameters
    ----------
    filename : str
        Filename of the output file to write the HEALPix map data
    hpmap : 1D `~numpy.ndarray`
        1D array containing the HEALPix map data, and the ordering
        scheme should be "RING";
        The data type is preserved in the output FITS file.
    header : `~astropy.io.fits.Header`, optional
        Extra header to be appended to the output FITS
    clobber : bool, optional
        Whether overwrite the existing file?
    checksum : bool, optional
        Whether calculate the checksum for the output file, which is
        recorded as the "CHECKSUM" header keyword.

    NOTE
    ----
    - This function is intended to replace the most common case of
      `healpy.write_map()`, which still uses some deprecated functions of
      `numpy` and `astropy`, meanwhile, it interface/arguments is not very
      handy.
    - This function (currently) only implement the very basic feature of
      the `healpy.write_map()`.
    """
    hpmap = np.asarray(hpmap)
    if hpmap.ndim != 1:
        raise ValueError("Invalid HEALPix data: only support 1D array")
    # Hack to ignore the dtype byteorder, use native endianness
    dtype = np.dtype(hpmap.dtype.type)
    hpmap = hpmap.astype(dtype)
    #
    npix = hpmap.size
    nside = int((npix / 12) ** 0.5)
    colfmt = FITS_COLUMN_FORMATS.get(hpmap.dtype)
    if hpmap.size > 1024:
        hpmap = hpmap.reshape(int(hpmap.size/1024), 1024)
        colfmt = "1024" + colfmt
    #
    hdr = fits.Header()
    # set HEALPix parameters
    hdr["PIXTYPE"] = ("HEALPIX", "HEALPix pixelization")
    hdr["ORDERING"] = ("RING",
                       "Pixel ordering scheme, either RING or NESTED")
    hdr["COORDSYS"] = ("G", "Ecliptic, Galactic or Celestial (equatorial)")
    hdr["NSIDE"] = (nside, "HEALPix resolution parameter")
    hdr["NPIX"] = (npix, "Total number of pixels")
    hdr["FIRSTPIX"] = (0, "First pixel # (0 based)")
    hdr["LASTPIX"] = (npix-1, "Last pixel # (0 based)")
    hdr["INDXSCHM"] = ("IMPLICIT", "Indexing: IMPLICIT or EXPLICIT")
    hdr["OBJECT"] = ("FULLSKY", "Sky coverage, either FULLSKY or PARTIAL")
    #
    hdr["EXTNAME"] = ("HEALPIX", "Name of the binary table extension")
    hdr["CREATOR"] = (__name__, "File creator")
    hdr["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                   "File creation date")
    # Merge user-provided header
    # NOTE: use the `.extend()` method instead of `.update()` method
    if header is not None:
        hdr.extend(fits.Header(header))
    #
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="I", array=hpmap, format=colfmt)
    ], header=hdr)
    hdu.writeto(filename, clobber=clobber, checksum=checksum)
