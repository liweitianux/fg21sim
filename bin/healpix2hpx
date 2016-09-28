#!/usr/bin/env python3
#
# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Reorganize the sky map in HEALPix table format into image in HPX projection.
"""


import os
import sys
import argparse

import numpy as np
from astropy.io import fits

import fg21sim
from fg21sim.utils import healpix2hpx


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize the HEALPix data to image in HPX projection")
    parser.add_argument("infile", help="input HEALPix data file")
    parser.add_argument("outfile", help="output FITS image in HPX projection")
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

    hpx_data, hpx_header = healpix2hpx(args.infile,
                                       append_history=history,
                                       append_comment=comments)
    if args.float:
        hpx_data = hpx_data.astype(np.float32)
    hdu = fits.PrimaryHDU(data=hpx_data, header=hpx_header)
    hdu.writeto(args.outfile, clobber=args.clobber, checksum=True)


if __name__ == "__main__":
    main()
