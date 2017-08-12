# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Input/output utilities.
"""

import os
import logging
import pickle
from datetime import datetime

import pandas as pd


logger = logging.getLogger(__name__)


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
    Save the given Pandas DataFrame into a CSV text file.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        The DataFrame to be saved to the CSV text file.
    outfile : string
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
        comment = ["by %s" % __name__,
                   "at %s" % datetime.now().isoformat()]

    with open(outfile, "w") as fh:
        # Write header comments with ``#`` prefixed.
        fh.write("".join(["# "+line.strip()+"\n" for line in comment]))
        df.to_csv(fh, header=True, index=False)
    logger.info("Wrote DataFrame to CSV file: {0}".format(outfile))


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
