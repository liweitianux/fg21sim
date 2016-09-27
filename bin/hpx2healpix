#!/usr/bin/env python3
#
# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Recover the sky map in HPX projection format back into HEALPix table format,
i.e., the reverse of `healpix2hpx.py`.
"""

import os
import sys
import argparse

import numpy as np
from astropy.io import fits

import fg21sim
from fg21sim.utils import hpx2healpix


# Reference:
# http://docs.astropy.org/en/stable/io/fits/usage/table.html#column-creation
FITS_COLUMN_FORMATS = {
    np.dtype("int16"): "I",
    np.dtype("int32"): "J",
    np.dtype("int64"): "K",
    np.dtype("float32"): "E",
    np.dtype("float64"): "D",
}


def main():
    parser = argparse.ArgumentParser(
        description="Recover the image in HPX projection to HEALPix data")
    parser.add_argument("infile", help="input FITS image in HPX projection")
    parser.add_argument("outfile", help="output HEALPix data file")
    parser.add_argument("-C", "--clobber", action="store_true",
                        help="overwrite the existing output file")
    parser.add_argument("-F", "--float", action="store_true",
                        help="use float (single precision) instead of double")
    args = parser.parse_args()

    tool = os.path.basename(sys.argv[0])
    history = [
        "TOOL: {0}".format(tool),
        "PARAM: {0}".format(" ".join(sys.argv[1:])),
    ]
    comments = [
        'Tool "{0}" is part of the "{1}" package'.format(tool,
                                                         fg21sim.__title__),
        'distributed under {0} license.'.format(fg21sim.__license__),
        'See also {0}'.format(fg21sim.__url__)
    ]

    hp_data, hp_header = hpx2healpix(args.infile,
                                     append_history=history,
                                     append_comment=comments)
    if args.float:
        hp_data = hp_data.astype(np.float32)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="I", array=hp_data,
                    format=FITS_COLUMN_FORMATS.get(hp_data.dtype))
    ], header=hp_header)
    hdu.writeto(args.outfile, clobber=args.clobber, checksum=True)


if __name__ == "__main__":
    main()
