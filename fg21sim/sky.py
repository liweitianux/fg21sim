# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Generic simulation sky supporting both sky patch and HEALPix all-sky
maps.
"""

import os
import logging
import copy

import numpy as np
from scipy import ndimage
from astropy.io import fits
import astropy.units as au
from astropy.coordinates import SkyCoord
from regions import PixCoord, RectanglePixelRegion
from reproject import reproject_interp, reproject_to_healpix
import healpy as hp

from .utils.wcs import make_wcs
from .utils.fits import read_fits_healpix, write_fits_healpix


logger = logging.getLogger(__name__)


class SkyPatch:
    """
    Support reading & writing FITS file of sky patches.

    NOTE/XXX
    --------
    Only `TAN` (tangential) sky projection supported!

    Parameters
    ----------
    size : (xsize, ysize) tuple
        The (pixel) dimensions of the (output) sky patch.
        If the input sky has a different size, then it will be *scaled*
        to match this output size.
        NOTE: Due to the FITS using Fortran ordering, while Python/numpy
              using C ordering, therefore, the read in image/data array
              has shape (ysize, xsize).
    pixelsize : float
        The pixel size of the sky patch, will be used to determine
        the sky coordinates. [ arcmin ]
    center : (ra, dec) tuple
        The (R.A., Dec.) coordinate of the sky patch center. [ deg ]
    infile : str
        The path to the input sky patch

    Attributes
    ----------
    type : str, "patch" or "healpix"
        The type of this sky map
    data : 1D `numpy.ndarray`
        The flattened 1D array of map data
        NOTE: The 2D image is flattened to 1D, making it easier to be
              manipulated in a similar way as the HEALPix map.
    size : int tuple, (width, height)
        The dimensions of the FITS image
    shape : int tuple, (nrow*ncol, )
        The shape of the flattened image array
        NOTE: nrow=height, ncol=width
    pixelsize : float
        The pixel size of the sky map [ arcmin ]
    center : float tuple, (ra, dec)
        The (R.A., Dec.) coordinate of the sky patch center. [ deg ]
    """
    type_ = "patch"
    # Input sky patch and its frequency [ MHz ]
    infile = None
    frequency = None
    # Sky data; should be a 1D ``numpy.ndarray`` (i.e., flattened)
    data = None
    # Coordinates of each pixel
    coordinates = None

    def __init__(self, size, pixelsize, center=(0.0, 0.0),
                 infile=None, frequency=None):
        self.xcenter, self.ycenter = center
        self.xsize, self.ysize = size
        self.pixelsize = pixelsize
        if infile is not None:
            self.read(infile, frequency)

    @property
    def size(self):
        return (self.xsize, self.ysize)

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
        else:
            return (self.ysize * self.xsize, )

    @property
    def center(self):
        return (self.xcenter, self.ycenter)

    def read(self, infile, frequency=None):
        """
        Read input sky data from file.

        Parameters
        ----------
        infile : str
            The path to the input sky patch
        frequency : float, optional
            The frequency of the sky patch; [ MHz ]
        """
        self.infile = infile
        if frequency is not None:
            self.frequency = frequency * au.MHz
        with fits.open(infile) as f:
            self.data = f[0].data
            self.header = f[0].header
        self.ysize_in, self.xsize_in = self.data.shape
        logger.info("Read sky patch from: %s (%dx%d)" %
                    (infile, self.xsize_in, self.ysize_in))
        if (self.xsize_in != self.xsize) or (self.ysize_in != self.ysize):
            logger.warning("Scale input sky patch to size %dx%d" %
                           (self.xsize, self.ysize))
            zoom = (self.ysize/self.ysize_in, self.xsize/self.xsize_in)
            self.data = ndimage.zoom(self.data, zoom=zoom, order=1)
        # Flatten the image
        self.data = self.data.flatten()
        logger.info("Flatten the image to a 1D array")

    def load(self, infile, frequency=None):
        """
        Make a new copy of this instance, then read the input sky patch
        and return the loaded new instance.

        Returns
        -------
        A new copy of this instance with the given sky patch loaded.
        """
        sky = self.copy()
        sky.read(infile=infile, frequency=frequency)
        return sky

    def copy(self):
        """
        Return a copy of this instance.
        """
        return copy.deepcopy(self)

    def write(self, outfile, clobber=False, checksum=True):
        """
        Write current data to file.
        """
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            logger.info("Created output directory: %s" % outdir)
        image = self.data.reshape(self.ysize, self.xsize)
        header = self.wcs.to_header()
        header.extend(self.header, update=True)
        hdu = fits.PrimaryHDU(data=image, header=self.header)
        hdu.writeto(outfile, clobber=clobber, checksum=checksum)
        logger.info("Write sky map to file: %s" % outfile)

    @property
    def wcs(self):
        """
        The WCS header with sky projection information, for sky <->
        pixel coordinate(s) conversion.

        NOTE/XXX
        --------
        Currently only support the `TAN` (tangential) projection.
        """
        w = make_wcs(center=(self.xcenter, self.ycenter),
                     size=(self.xsize, self.ysize),
                     pixelsize=self.pixelsize,
                     frame="ICRS", projection="TAN")
        return w

    def contains(self, skycoord):
        """
        Check whether the given (list of) sky coordinate(s) falls
        inside this sky patch (region).

        Parameters
        ----------
        skycoord : `~astropy.coordinate.SkyCoord` or (lon, lat) tuple
            The (list of) sky coordinate(s) to check, or the (list of)
            longitudes and latitudes of sky coordinates [ deg ]

        Returns
        -------
        inside : bool
            (list of) boolean values indicating whether the given
            sky coordinate(s) is inside this sky patch.
        """
        if not isinstance(skycoord, SkyCoord):
            lon, lat = skycoord
            skycoord = SkyCoord(lon, lat, unit=au.deg)
        wcs = self.wcs
        pixcoord = PixCoord.from_sky(skycoord, wcs=wcs)
        center = PixCoord(x=self.xcenter, y=self.ycenter)
        region = RectanglePixelRegion(center=center,
                                      width=self.xsize, height=self.ysize)
        return region.contains(pixcoord)

    def reproject_from(self, data, wcs, squeeze=False):
        """
        Reproject the given image/data together with WCS information
        onto the grid of this sky.

        Parameters
        ----------
        data : 2D float `~numpy.ndarray`
            The input data/image to be reprojected
        wcs : `~astropy.wcs.WCS`
            The WCS information of the input data/image (naxis=2)
        squeeze : bool, optional
            Whether to squeeze the reprojected data to only keep
            the positive pixels.

        Returns
        -------
        If ``squeeze=True``, then returns tuple of ``(indexes, values)``,
        otherwise, returns the reprojected image/data array.

        indexes : 1D int `~numpy.ndarray`
            The indexes of the pixels with positive values.
        values : 1D float `~numpy.ndarray`
            The values of the above pixels.

        reprojected : 1D `~numpy.ndarray`
            The reprojected data/image with same shape of this sky,
            i.e., ``self.data``.
        """
        eps = 1e-5
        wcs_out = self.wcs
        shape_out = (self.ysize, self.xsize)
        reprojected, __ = reproject_interp(
            input_data=(data, wcs), output_projection=wcs_out,
            shape_out=shape_out)
        reprojected = reprojected.flatten()
        if squeeze:
            with np.errstate(invalid="ignore"):
                indexes = reprojected > eps
            values = reprojected[indexes]
            return (indexes, values)
        else:
            return reprojected


class SkyHealpix:
    """
    Support the HEALPix all-sky map.

    Parameters
    ----------
    nside : int
        The pixel resolution of HEALPix (must be power of 2)

    Attributes
    ----------
    shape : int tuple, (npix,)
        The shape (i.e., length) of the HEALPix array
    pixelsize : float
        The pixel size of the HEALPix map [ arcmin ]
    """
    type_ = "healpix"
    # Input sky patch and its frequency [ MHz ]
    infile = None
    frequency = None
    # Sky data; should be a ``numpy.ndarray``
    data = None
    # Coordinates of each pixel
    coordinates = None

    def __init__(self, nside, infile=None, frequency=None):
        self.nside = nside
        if infile is not None:
            self.read(infile, frequency)

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
        else:
            return (hp.nside2npix(self.nside), )

    @property
    def pixelsize(self):
        return hp.nside2resol(self.nside, arcmin=True)

    def read(self, infile, frequency=None):
        """
        Read input HEALPix all-sky map.

        Parameters
        ----------
        infile : str
            The path to the input HEALPix all-sky map.
        frequency : float, optional
            The frequency of the sky patch; [ MHz ]
        """
        self.infile = infile
        if frequency is not None:
            self.frequency = frequency * au.MHz
        self.data, self.header = read_fits_healpix(infile)
        self.nside_in = self.header["NSIDE"]
        logger.info("Read HEALPix sky map from: {0} (Nside={1})".format(
            infile, self.nside_in))
        if self.nside_in != self.nside:
            self.data = hp.ud_grade(self.data, nside_out=self.nside)
            logger.warning("Upgrade/downgrade sky map from Nside " +
                           "{0} to {1}".format(self.nside_in, self.nside))

    def load(self, infile, frequency=None):
        """
        Make a new copy of this instance, then read the input sky map
        and return the loaded new instance.

        Returns
        -------
        A new copy of this instance with the given sky map loaded.
        """
        sky = self.copy()
        sky.read(infile=infile, frequency=frequency)
        return sky

    def copy(self):
        """
        Return a copy of this instance.
        """
        return copy.deepcopy(self)

    def write(self, outfile, clobber=False, checksum=True):
        """
        Write current data to file.
        """
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            logger.info("Created output directory: %s" % outdir)
        write_fits_healpix(outfile, self.data, header=self.header,
                           clobber=clobber, checksum=checksum)
        logger.info("Write sky map to file: %s" % outfile)

    def contains(self, skycoord):
        """
        Shim method to be consistent with ``SkyPatch``.

        Always returns ``True``, since the HEALPix map covers all sky.
        """
        if skycoord.isscalar:
            return True
        else:
            return np.ones(shape=len(skycoord), dtype=np.bool)

    def reproject_from(self, data, wcs, squeeze=False):
        """
        Reproject the given image/data together with WCS information
        onto the grid of this sky.

        Parameters
        ----------
        data : 2D float `~numpy.ndarray`
            The input data/image to be reprojected
        wcs : `~astropy.wcs.WCS`
            The WCS information of the input data/image (naxis=2)
        squeeze : bool, optional
            Whether to squeeze the reprojected data to only keep
            the positive pixels.

        Returns
        -------
        If ``squeeze=True``, then returns tuple of ``(indexes, values)``,
        otherwise, returns the reprojected image/data array.

        indexes : 1D int `~numpy.ndarray`
            The indexes of the pixels with positive values.
        values : 1D float `~numpy.ndarray`
            The values of the above pixels.

        reprojected : 1D `~numpy.ndarray`
            The reprojected data/image with same shape of this sky,
            i.e., ``self.data.shape``.
        """
        eps = 1e-5
        reprojected, __ = reproject_to_healpix(
            input_data=(data, wcs), coord_system_out="galactic",
            nested=False, nside=self.nside)
        if squeeze:
            with np.errstate(invalid="ignore"):
                indexes = reprojected > eps
            values = reprojected[indexes]
            return (indexes, values)
        else:
            return reprojected


def get_sky(configs):
    """
    Sky class factory function to support both the sky patch and
    HEALPix all-sky map.

    Parameters
    ----------
    configs : ConfigManager object
        An `ConfigManager` object contains default and user configurations.
        For more details, see the example config specification.
    """
    skytype = configs.getn("sky/type")
    if skytype == "patch":
        sec = "sky/patch"
        xsize = configs.getn(sec+"/xsize")
        ysize = configs.getn(sec+"/ysize")
        xcenter = configs.getn(sec+"/xcenter")
        ycenter = configs.getn(sec+"/ycenter")
        pixelsize = configs.getn(sec+"/pixelsize")
        return SkyPatch(size=(xsize, ysize), pixelsize=pixelsize,
                        center=(xcenter, ycenter))
    elif skytype == "healpix":
        sec = "sky/healpix"
        nside = configs.getn(sec+"/nside")
        return SkyHealpix(nside=nside)
    else:
        raise ValueError("unknown sky type: %s" % skytype)
