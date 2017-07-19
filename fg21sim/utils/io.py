# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Input/output utilities.
"""

import os
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def dataframe_to_csv(df, outfile, clobber=False):
    """
    Save the given Pandas DataFrame into a CSV text file.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        The DataFrame to be saved to the CSV text file.
    outfile : string
        The path to the output CSV file.
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
            logger.warning("Remove existing file: {0}".format(outfile))
            os.remove(outfile)
        else:
            raise OSError("Output file exists: {0}".format(outfile))
    df.to_csv(outfile, header=True, index=False)
    logger.info("Saved DataFrame to CSV file: {0}".format(outfile))
