# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Input/output utilities.
"""

import os
import logging
from datetime import datetime

import pandas as pd


logger = logging.getLogger(__name__)


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

    # Create directory if necessary
    dirname = os.path.dirname(outfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info("Created output directory: {0}".format(dirname))

    if os.path.exists(outfile):
        if clobber:
            logger.warning("Removed existing file: {0}".format(outfile))
            os.remove(outfile)
        else:
            raise OSError("Output file exists: {0}".format(outfile))

    # Add a default header comment
    if comment is None:
        comment = ["by %s" % __name__,
                   "at %s" % datetime.now().isoformat()]

    with open(outfile, "w") as fh:
        # Write header comments with ``#`` prefixed.
        fh.write("".join(["# "+line.strip()+"\n" for line in comment]))
        df.to_csv(fh, header=True, index=False)
    logger.info("Wrote DataFrame to CSV file: {0}".format(outfile))
