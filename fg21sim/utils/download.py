# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Utilities to download files
"""

import os
import subprocess
import logging

from .hashutil import calc_md5


logger = logging.getLogger(__name__)


def _check_filesize(filepath, size):
    """
    Check whether the file size equals the given size.

    Parameters
    ----------
    filepath : str
        Path to the input file
    size : int
        Expected/true size (in bytes) of the file

    Returns
    -------
    valid : bool
        ``True`` if the on-disk file size equals the given size,
        otherwise ``False``.
    """
    size_ondisk = os.path.getsize(filepath)
    return size_ondisk == size


def _check_md5(filepath, md5):
    """
    Check whether the file MD5 digest matches the given MD5.

    Parameters
    ----------
    filepath : str
        Path to the input file
    md5 : str
        Expected/true MD5 digest of the file

    Returns
    -------
    valid : bool
        ``True`` if the on-disk file MD5 digest matches the given MD5,
        otherwise ``False``.
    """
    md5_ondisk = calc_md5(filepath)
    return md5_ondisk == md5


def download_file(url, outfile=None, size=None, md5=None, clobber=False):
    """
    Download file using "wget" and validate the file size and MD5 digest.

    If the expected MD5 digest is provided, and the on-disk file has the
    same MD5 digest, then the download is skipped.

    If the output file already exists but with unmatched MD5, then
    re-download it if ``clobber=True``, otherwise, an ``IOError`` raised.

    Parameters
    ----------
    url : str
        The URL from where to download the file.
    outfile : str, optional
        The path and filename for the downloaded file.
        If not provided, then use the basename of the URL.
    size : int, optional
        Expected/true size (in bytes) of the file
        If provided, then check the file size after download.
    md5 : str, optional
        Expected/true MD5 digest of the file
        If provided, then check the MD5 digest after download.
    clobber : bool, optional
        Whether to overwrite the existing file?

    Raises
    ------
    IOError :
        Output file with unmatched MD5 digest already exists
        while ``clobber=False``.
    """
    if outfile is None:
        outfile = os.path.basename(url)
    # Check whether can skip the download
    if os.path.exists(outfile):
        if (md5 is not None) and _check_md5(outfile, md5):
            logger.info("Skip already downloaded file: {0}".format(outfile))
            return
        elif clobber:
            os.remove(outfile)
            logger.info("Removed wrong existing file: {0}".format(outfile))
        else:
            raise IOError("Exists wrong output file: {0}".format(outfile))
    #
    cmd = ["wget", "-O", outfile, url]
    logger.info("CMD: {0}".format(" ".join(cmd)))
    subprocess.check_call(cmd)
    if (size is not None) and (not _check_filesize(outfile, size)):
        raise ValueError("Downloaded file has wrong size")
    if (md5 is not None) and (not _check_md5(outfile, md5)):
        raise ValueError("Downloaded file has unmatched MD5 digest")
    logger.info("Downloaded and validated file: {0}".format(outfile))
