# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Hash utilities.

md5 :
    Calculate the MD5 checksum of the file.
"""

import hashlib


def calc_md5(filepath, blocksize=65536):
    """
    Calculate the MD5 checksum/digest of the file data.

    Parameters
    ----------
    filepath : str
        The path to the file
    blocksize : int, optional
        The block size (bytes) of chunks when read the file.

    Returns
    -------
    digest : str
        The checksum/digest of the file data, containing only hexadecimal
        digits.

    Credits
    -------
    - Stackoverflow: Generating an MD5 checksum of a file
      https://stackoverflow.com/a/3431838/4856091
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
