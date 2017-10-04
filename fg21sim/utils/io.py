# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Input/output utilities
----------------------
* dataframe_to_csv:
  Save the given Pandas DataFrame into a CSV text file.

* pickle_dump:
  Dump the given object into the output file using ``pickle.dump()``.

* pickle_load:
  Load the pickled Python back from the given file.

* write_fits_image:
  Write the supplied image (together with header information) into
  the output FITS file.

* read_fits_healpix:
  Read the HEALPix map from a FITS file or a BinTableHDU to 1D array
  in *RING* ordering.

* write_fits_healpix:
  Write the HEALPix map to a FITS file with proper header as well
  as the user-provided header.
"""

import os
import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.io import fits
import healpy as hp


logger = logging.getLogger(__name__)


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


def _create_dir(filepath):
    """
    Check the existence of the target directory, and create it if necessary.

    NOTE
    ----
    If the given ``filepath`` is simply the filename without any directory
    path, then just returns.
    """
    dirname = os.path.dirname(filepath)
    # ``dirname == ""`` if ``filepath`` does not contain directory path
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info("Created output directory: {0}".format(dirname))


def _check_existence(filepath, clobber=False, remove=False):
    """
    Check the existence of the target file.

    * raise ``OSError`` : file exists and clobber is False;
    * no action : files does not exists or clobber is True;
    * remove the file : files exists and clobber is True and remove is True
    """
    if os.path.exists(filepath):
        if clobber:
            if remove:
                logger.warning("Removed existing file: {0}".format(filepath))
                os.remove(filepath)
            else:
                logger.warning("Existing file will be overwritten.")
        else:
            raise OSError("Output file exists: {0}".format(filepath))


def dataframe_to_csv(df, outfile, comment=None, clobber=False):
    """
    Save the given Pandas DataFrame into a CSV text file with comments
    prepended at the file head.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        The DataFrame to be saved to the CSV text file.
    outfile : str
        The path to the output CSV file.
    comment : list[str], optional
        A list of comments to be prepended to the output CSV file header.
        The prefix ``#`` is not required and will be automatically added.
    clobber : bool, optional
        Whether overwrite the existing output file?
        Default: False
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Not a Pandas DataFrame!")

    _create_dir(outfile)
    _check_existence(outfile, clobber=clobber, remove=True)

    # Add a default header comment
    if comment is None:
        comment = [
            "by %s" % __name__,
            "at %s" % datetime.now(timezone.utc).astimezone().isoformat(),
        ]

    with open(outfile, "w") as fh:
        # Write header comments with ``#`` prefixed.
        fh.write("".join(["# "+line.strip()+"\n" for line in comment]))
        df.to_csv(fh, header=True, index=False)
    logger.info("Wrote DataFrame to CSV file: {0}".format(outfile))


def csv_to_dataframe(infile):
    """
    Read the given CSV file as a Pandas DataFrame, with head comments
    also considered and returned.

    Parameters
    ----------
    infile : str
        The path to the input CSV file.

    Returns
    df : `~pandas.DataFrame`
        The DataFrame read from the CSV text file.
    comment : list[str]
        A list of comments read from the lines prefixing with ``#``
        at the CSV file header.
        The prefix ``#`` is striped.
    """
    comments = []
    for line in open(infile):
        line = line.strip()
        if line == "":
            continue
        elif line[0] == "#":
            comments.append(line.lstrip("# "))
        else:
            break

    df = pd.read_csv(infile, comment="#")
    return (df, comments)


def pickle_dump(obj, outfile, clobber=False):
    """
    Dump the given object into the output file using ``pickle.dump()``.

    NOTE
    ----
    The dumped output file is in binary format, and can be loaded back
    using ``pickle.load()``, e.g., the ``pickle_load()`` function below.

    Example
    -------
    >>> a = [1, 2, 3]
    >>> pickle.dump(a, file=open("a.pkl", "wb"))
    >>> b = pickle.load(open("a.pkl", "rb))
    >>> a == b
    True

    Parameters
    ----------
    outfile : str
        The path/filename to the output file storing the pickled object.
    clobber : bool, optional
        Whether to overwrite the existing output file.
        Default: False
    """
    _create_dir(outfile)
    _check_existence(outfile, clobber=clobber, remove=True)
    pickle.dump(obj, file=open(outfile, "wb"))
    logger.info("Pickled data to file: %s" % outfile)


def pickle_load(infile):
    """
    Load the pickled Python back from the given file.

    Parameters
    ----------
    infile : str
        The path/filename to the data file, e.g., dumped by the above
        ``pickle_dump()`` function.

    Returns
    -------
    obj : The loaded Python object from the input file.
    """
    return pickle.load(open(infile, "rb"))


def write_fits_image(outfile, image, header=None, float32=False,
                     clobber=False, checksum=False):
    """
    Write the supplied image (together with header information) into
    the output FITS file.

    Parameters
    ----------
    outfile : str
        The path/filename to the output file storing the pickled object.
    image : 2D `~numpy.ndarray`
        The image data to be written out to the FITS file.
        NOTE: image.shape: (nrow, ncol)  <->  FITS image: (ncol, nrow)
    header : `~astropy.io.fits.Header`
        The FITS header information for this image
    float32 : bool, optional
        Whether coerce the image data (generally double/float64 data type)
        into single/float32 (in order to save space)?
        Default: False (i.e., preserve the data type unchanged)
    clobber : bool, optional
        Whether to overwrite the existing output file.
        Default: False
    checksum : bool, optional
        Whether to calculate the data checksum, which may cost some time?
        Default: False
    """
    _create_dir(outfile)
    _check_existence(outfile, clobber=clobber, remove=True)

    hdr = fits.Header()
    hdr["CREATOR"] = (__name__, "File creator")
    hdr["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                   "File creation date")
    if header is not None:
        hdr.extend(header, update=True)

    if float32:
        image = np.asarray(image, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=image, header=header)
    hdu.writeto(outfile, checksum=checksum)
    logger.info("Wrote image to FITS file: %s" % outfile)


def read_fits_healpix(filename):
    """
    Read the HEALPix map from a FITS file or a BinTableHDU to 1D array
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
    the header of input FITS file.
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


def write_fits_healpix(outfile, hpmap, header=None, float32=False,
                       clobber=False, checksum=False):
    """
    Write the HEALPix map to a FITS file with proper header as well
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
    outfile : str
        Filename of the output file to write the HEALPix map data
    hpmap : 1D `~numpy.ndarray`
        1D array containing the HEALPix map data, and the ordering
        scheme should be "RING";
        The data type is preserved or cast into single/float32 if the
        below ``float32`` parameter is True, in the output FITS file.
    header : `~astropy.io.fits.Header`, optional
        Extra header to be appended to the output FITS
    float32 : bool, optional
        Whether coerce the image data (generally double/float64 data type)
        into single/float32 (in order to save space)?
        Default: False (i.e., preserve the data type unchanged)
    clobber : bool, optional
        Whether to overwrite the existing output file?
        Default: False
    checksum : bool, optional
        Whether to calculate the data checksum, which may cost some time?
        Default: False

    NOTE
    ----
    - This function is intended to replace the most common case of
      `healpy.write_map()`, which still uses some deprecated functions of
      `numpy` and `astropy`, meanwhile, its interface/arguments is not very
      handy.
    - This function (currently) only implement the very basic feature of
      the `healpy.write_map()`.
    """
    _create_dir(outfile)
    _check_existence(outfile, clobber=clobber, remove=True)

    hpmap = np.asarray(hpmap)
    if hpmap.ndim != 1:
        raise ValueError("Invalid HEALPix data: only support 1D array")
    if float32:
        dtype = np.float32
    else:
        # HACK: ignore the dtype byteorder, use native endianness
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
        hdr.extend(header, update=True)
    #
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="I", array=hpmap, format=colfmt)
    ], header=hdr)
    hdu.writeto(outfile, checksum=checksum)
    logger.info("Wrote HEALPix map to FITS file: %s" % outfile)


def write_dndlnm(outfile, dndlnm, z, mass, clobber=False):
    """
    Write the halo mass distribution data into file in NumPy's ".npz"
    format, which packs the ``dndlnm``, ``z``, and ``mass`` arrays.

    Parameters
    ----------
    outfile : str
        The output file to store the dndlnm data, in ".npz" format.
    dndlnm : 2D float `~numpy.ndarray`
        Shape: (len(z), len(mass))
        Differential mass function in terms of natural log of M.
        Unit: [Mpc^-3] (the little "h" is folded into the values)
    z : 1D float `~numpy.ndarray`
        Redshifts where the halo mass distribution is calculated.
    mass : 1D float `~numpy.ndarray`
        (Logarithmic-distributed) masses points.
        Unit: [Msun] (the little "h" is folded into the values)
    clobber : bool, optional
        Whether to overwrite the existing output file?
    """
    _create_dir(outfile)
    _check_existence(outfile, clobber=clobber, remove=True)
    np.savez(outfile, dndlnm=dndlnm, z=z, mass=mass)


def read_dndlnm(infile):
    """
    Read the halo mass distribution data from the above saved file.

    Parameters
    ----------
    infile : str
        The ".npz" file from which to read the dndlnm data.

    Returns
    -------
    (dndlnm, z, mass)
    """
    with np.load(infile) as npzfile:
        dndlnm = npzfile["dndlnm"]
        z = npzfile["z"]
        mass = npzfile["mass"]
    return (dndlnm, z, mass)
