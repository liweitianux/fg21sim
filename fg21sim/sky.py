# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Generic simulation sky supporting both sky patch and HEALPix all-sky
maps.

References
----------
* Python - 3. Data Model
  https://docs.python.org/3/reference/datamodel.html#special-method-names
"""

import logging
import copy
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage
from astropy.io import fits
import astropy.units as au
from astropy.coordinates import SkyCoord
from regions import PixCoord, RectanglePixelRegion
from reproject import reproject_interp, reproject_to_healpix
import healpy as hp

from .utils.wcs import make_wcs
from .utils.io import (read_fits_healpix,
                       write_fits_healpix,
                       write_fits_image)
from .utils.random import spherical_uniform
from .utils.units import UnitConversions as AUC


logger = logging.getLogger(__name__)


class SkyBase:
    """
    The base class for both the sky patch and HEALPix all-sky
    map classes.

    Attributes
    ----------
    type_ : str
        The type of the sky image
        Values: ``patch`` or ``healpix``
    data : `~numpy.ndarray`
        The data array read from input sky image, or to be written into
        output FITS file.
    frequency_ : float
        The frequency of the input/output sky image.
        Unit: [MHz]
    creator_ : str
        The creator of the (output) sky image.
        Default: ``__name__``
    header_ : `~astropy.io.fits.Header`
        The FITS header information of the input/output file.
    float32_ : bool
        Whether to use single/float32 data type to save the sky image?
        Default: True
    clobber_ : bool, optional
        Whether to overwrite the existing output file.
        Default: False
    checksum_ : bool, optional
        Whether to calculate the checksum data for the output
        FITS file, which may cost some time.
        Default: False
    """
    def __init__(self, float32=True, clobber=False, checksum=False):
        self.type_ = None
        self.data = None
        self.frequency_ = None
        self.creator_ = __name__
        self.header_ = fits.Header()
        self.float32_ = float32
        self.clobber_ = clobber
        self.checksum_ = checksum

    def __add__(self, other):
        """Binary arithmetic operation: ``+``."""
        if isinstance(other, self.__class__):
            return self.data + other.data
        elif isinstance(other, (int, float, np.ndarray)):
            return self.data + other
        else:
            raise NotImplemented

    def __sub__(self, other):
        """Binary arithmetic operation: ``-``."""
        if isinstance(other, self.__class__):
            return self.data - other.data
        elif isinstance(other, (int, float, np.ndarray)):
            return self.data - other
        else:
            raise NotImplemented

    def __mul__(self, other):
        """Binary arithmetic operation: ``*``."""
        if isinstance(other, self.__class__):
            return self.data * other.data
        elif isinstance(other, (int, float, np.ndarray)):
            return self.data * other
        else:
            raise NotImplemented

    def __truediv__(self, other):
        """Binary arithmetic operation: ``/``."""
        if isinstance(other, self.__class__):
            return self.data / other.data
        elif isinstance(other, (int, float, np.ndarray)):
            return self.data / other
        else:
            raise NotImplemented

    def __pow__(self, other):
        """Binary arithmetic operation: ``**``."""
        if isinstance(other, self.__class__):
            return self.data ** other.data
        elif isinstance(other, (int, float, np.ndarray)):
            return self.data ** other
        else:
            raise NotImplemented

    def __neg__(self):
        """Unary arithmetic operation: ``-``."""
        return -self.data

    def __abs__(self):
        """Unary arithmetic operation: ``abs()``."""
        return np.abs(self.data)

    @property
    def shape(self):
        """
        Numpy array shape of the (current/output) sky data.
        """
        return self.data.shape

    @property
    def frequency(self):
        """
        The frequency of the sky image.
        Unit: [MHz]
        """
        if self.frequency_ is not None:
            return self.frequency_
        else:
            return self.header_.get("FREQ", None)

    @frequency.setter
    def frequency(self, value):
        """
        Set the frequency of the sky image.
        Unit: [MHz]
        """
        self.frequency_ = value

    @property
    def creator(self):
        """
        The creator of the sky image.
        """
        if self.creator_ is not None:
            return self.creator_
        else:
            return self.header_.get("CREATOR", None)

    @creator.setter
    def creator(self, value):
        """
        Set the creator of the sky image.
        """
        self.creator_ = value

    @property
    def header(self):
        """
        The FITS header of the current sky.
        """
        hdr = self.header_.copy()
        hdr["SkyType"] = (self.type_, "Patch / HEALPix")
        hdr["PixSize"] = (self.pixelsize, "Pixel size [arcsec]")
        hdr["CREATOR"] = (self.creator, "Sky Creator")
        hdr["DATE"] = (datetime.now(timezone.utc).astimezone().isoformat(),
                       "File creation date")
        return hdr

    def add_header(self, key, value, comment=None):
        """
        Add/update a key to the FITS header.
        """
        if comment is None:
            self.header_[key] = value
        else:
            self.header_[key] = (value, comment)

    def copy(self):
        """
        Return a (deep) copy of this instance.
        """
        return copy.deepcopy(self)

    def load(self, infile, frequency=None):
        """
        Make a new *copy* of this instance, then read the given sky
        image in and return the loaded new instance.
        """
        sky = self.copy()
        sky.read(infile=infile, frequency=frequency)
        return sky

    def read(self, infile, frequency=None):
        """
        Read the given sky image into this instance.

        Parameters
        ----------
        infile : str
            The path to the given input sky image.
        frequency : float, optional
            The frequency of the  given sky image if applicable.
            Unit: [MHz]
        """
        raise NotImplementedError

    def write(self, outfile, clobber=None):
        """
        Write the sky image (with current data) into a FITS file.

        Parameters
        ----------
        outfile : str
            The path/filename to the output FITS file.
        clobber : bool, optional
            If not ``None``, then overwrite the default ``self.clobber_``
            from the configuration file, to determine whether to overwrite
            the existing output file.
        """
        raise NotImplementedError

    @property
    def area(self):
        """
        Sky coverage of the sky.
        Unit: [deg^2]
        """
        raise NotImplementedError

    @property
    def pixelsize(self):
        """
        Pixel size of the sky image.
        Unit: [arcsec]
        """
        raise NotImplementedError

    def random_points(self, n=1):
        """
        Generate uniformly distributed random points within the
        sky image (coverage).

        Parameters
        ----------
        n : int, optional
            The number of random points required.
            Default: 1

        Returns
        -------
        lon, lat : float, or 1D `~numpy.ndarray`
            The longitudes and latitudes (in world coordinate)
            generated.
            Unit: [deg]
        """
        raise NotImplementedError


class SkyPatch(SkyBase):
    """
    Support reading & writing FITS file of sky patches.

    NOTE/XXX
    --------
    Currently just use ``CAR`` (Cartesian) sky projection, i.e.,
    assuming a flat sky!

    Parameters
    ----------
    size : (xsize, ysize) tuple
        The (pixel) dimensions of the (output) sky patch.
        If the input sky has a different size, then it will be *scaled*
        to match this output size.
        NOTE: Due to the FITS using Fortran ordering, while Python/numpy
              using C ordering, therefore, the read in image/data array
              has shape ``(ysize, xsize)``, therefore, the ``self.data``
              should be reshaped to ``(ysize, xsize)`` on output.
    pixelsize : float
        The pixel size of the sky patch, will be used to determine
        the sky coordinates.
        Unit: [arcsec]
    center : (ra, dec) tuple, optional
        The (R.A., Dec.) coordinate of the sky patch center.
        Unit: [deg]
    infile : str, optional
        The path to the input sky patch
    frequency : float, optional
        The frequency of the input sky path
        Unit: [MHz]

    Attributes
    ----------
    type_ : "patch"
        This is a sky patch.
    data : 2D `~numpy.ndarray`
        The 2D data array of sky image.
        (HEALPix map stores data in an 1D array.)
    """
    def __init__(self, size, pixelsize, center=(0.0, 0.0),
                 infile=None, frequency=None, **kwargs):
        super().__init__(**kwargs)

        self.type_ = "patch"
        self.xsize, self.ysize = size
        # Initialize an empty image
        self.data = np.zeros(shape=(self.ysize, self.xsize))
        self.pixelsize = pixelsize
        self.xcenter, self.ycenter = center

        if infile is not None:
            self.read(infile, frequency)

    @property
    def area(self):
        """
        The sky coverage of this patch.
        Unit: [deg^2]

        XXX/FIXME
        ---------
        Assumed a flat sky!
        Consider the spherical coordination and WCS sky projection!!
        """
        lonsize, latsize = self.size
        return (lonsize * latsize)

    @property
    def size(self):
        """
        The sky patch size along X/longitude and Y/latitude axes.

        Returns
        -------
        (lonsize, latsize) : float tuple
            Longitudinal and latitudinal sizes
            Unit: [deg]
        """
        return (self.xsize * self.pixelsize * AUC.arcsec2deg,
                self.ysize * self.pixelsize * AUC.arcsec2deg)

    @property
    def center(self):
        return (self.xcenter, self.ycenter)

    @property
    def lon_limit(self):
        """
        The longitudinal (X axis) limits.

        Returns
        -------
        (lon_min, lon_max) : float tuple
            The minimum and maximum longitudes (X axis).
            Unit: [deg]
        """
        lonsize, latsize = self.size
        return (self.xcenter - 0.5*lonsize,
                self.xcenter + 0.5*lonsize)

    @property
    def lat_limit(self):
        """
        The latitudinal (Y axis) limits.

        Returns
        -------
        (lat_min, lat_max) : float tuple
            The minimum and maximum latitudes (Y axis).
            Unit: [deg]
        """
        lonsize, latsize = self.size
        return (self.ycenter - 0.5*latsize,
                self.ycenter + 0.5*latsize)

    def read(self, infile, frequency=None):
        """
        Read input sky data from file.

        Parameters
        ----------
        infile : str
            The path to the input sky patch
        frequency : float, optional
            The frequency of the sky patch;
            Unit: [MHz]
        """
        self.infile = infile
        if frequency is not None:
            self.frequency = frequency
        with fits.open(infile) as f:
            self.data = f[0].data
            header = f[0].header.copy(strip=True)
            self.header_.extend(header, update=True)
        self.ysize_in, self.xsize_in = self.data.shape
        logger.info("Read sky patch from: %s (%dx%d)" %
                    (infile, self.xsize_in, self.ysize_in))

        if (self.xsize_in != self.xsize) or (self.ysize_in != self.ysize):
            logger.warning("Scale input sky patch to size %dx%d" %
                           (self.xsize, self.ysize))
            zoom = (self.ysize/self.ysize_in, self.xsize/self.xsize_in)
            self.data = ndimage.zoom(self.data, zoom=zoom, order=1)

    def write(self, outfile, clobber=None):
        """
        Write current data to file.
        """
        if clobber is None:
            clobber = self.clobber_
        write_fits_image(outfile, image=self.data, header=self.header,
                         float32=self.float32_,
                         clobber=clobber,
                         checksum=self.checksum_)

    @property
    def header(self):
        """
        FITS header of the sky for storing information in the output file.
        """
        hdr = super().header
        hdr.extend(self.wcs.to_header(), update=True)
        hdr["RA0"] = (self.center[0], "R.A. of patch center [deg]")
        hdr["DEC0"] = (self.center[1], "Dec. of patch center [deg]")
        return hdr

    @property
    def wcs(self):
        """
        The WCS header with sky projection information, for sky <->
        pixel coordinate(s) conversion.

        NOTE/XXX
        --------
        Currently just use the `CAR` (Cartesian) projection,
        i.e., assuming a flat sky.
        """
        w = make_wcs(center=(self.xcenter, self.ycenter),
                     size=(self.xsize, self.ysize),
                     pixelsize=self.pixelsize,
                     frame="ICRS", projection="CAR")
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

    def reproject_from(self, data, wcs, squeeze=False, eps=1e-5):
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
            the pixels greater than a small positive value specified
            by parameter ``eps``.
            Default: False
        eps : float, optional
            The small positive value to specify the squeeze threshold.
            Default: 1e-5

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

    def random_points(self, n=1):
        """
        Generate uniformly distributed random points within the sky patch.

        Returns
        -------
        lon : float, or 1D `~numpy.ndarray`
            Longitudes (Galactic/equatorial);
            Unit: [deg]
        lat : float, or 1D `~numpy.ndarray`
            Latitudes (Galactic/equatorial);
            Unit: [deg]
        """
        lon_min, lon_max = self.lon_limit
        lat_min, lat_max = self.lat_limit
        lon = np.random.uniform(low=lon_min, high=lon_max, size=n)
        lat = np.random.uniform(low=lat_min, high=lat_max, size=n)
        return (lon, lat)


class SkyHealpix(SkyBase):
    """
    Support the HEALPix all-sky map.

    Parameters
    ----------
    nside : int
        The pixel resolution of HEALPix (must be power of 2)
    infile : str, optional
        The path to the input sky patch
    frequency : float, optional
        The frequency of the input sky path
        Unit: [MHz]

    Attributes
    ----------
    shape : int tuple, (npix,)
        The shape (i.e., length) of the HEALPix array
    pixelsize : float
        The pixel size of the HEALPix map
        Unit: [arcsec]
    """
    def __init__(self, nside, infile=None, frequency=None, **kwargs):
        super().__init__(**kwargs)

        self.type_ = "healpix"
        self.nside = nside
        self.data = np.zeros(shape=hp.nside2npix(self.nside))

        if infile is not None:
            self.read(infile, frequency)

    @property
    def area(self):
        """
        The sky coverage of this HEALPix map, i.e., all sky = 4π,
        Unit: [deg^2]
        """
        return 4*np.pi * AUC.rad2deg**2

    @property
    def pixelsize(self):
        ps = hp.nside2resol(self.nside, arcmin=True)
        ps *= AUC.arcmin2arcsec
        return ps

    def read(self, infile, frequency=None):
        """
        Read input HEALPix all-sky map.

        Parameters
        ----------
        infile : str
            The path to the input HEALPix all-sky map.
        frequency : float, optional
            The frequency of the sky patch;
            Unit: [MHz]
        """
        self.infile = infile
        if frequency is not None:
            self.frequency = frequency
        self.data, header = read_fits_healpix(infile)
        self.header_.extend(header, update=True)
        self.nside_in = header["NSIDE"]
        logger.info("Read HEALPix sky map from: {0} (Nside={1})".format(
            infile, self.nside_in))
        if self.nside_in != self.nside:
            self.data = hp.ud_grade(self.data, nside_out=self.nside)
            logger.warning("Upgrade/downgrade sky map from Nside " +
                           "{0} to {1}".format(self.nside_in, self.nside))

    def write(self, outfile, clobber=None):
        """
        Write current data to file.
        """
        if clobber is None:
            clobber = self.clobber_
        write_fits_healpix(outfile, hpmap=self.data, header=self.header,
                           float32=self.float32_,
                           clobber=self.clobber_,
                           checksum=self.checksum_)

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

    def random_points(self, n=1):
        """
        Generate uniformly distributed random points within the sky
        (i.e., all sky; on an unit sphere).

        Returns
        -------
        lon : float, or 1D `~numpy.ndarray`
            Longitudes (Galactic/equatorial), [0, 360) [deg].
        lat : float, or 1D `~numpy.ndarray`
            Latitudes (Galactic/equatorial), [-90, 90] [deg].
        """
        theta, phi = spherical_uniform(n)
        lon = np.degrees(phi)
        lat = 90.0 - np.degrees(theta)
        return (lon, lat)


##########################################################################


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
    # Parameters for the base sky class
    kwargs = {
        "float32": configs.getn("output/use_float"),
        "clobber": configs.getn("output/clobber"),
        "checksum": configs.getn("output/checksum"),
    }

    skytype = configs.getn("sky/type")
    if skytype == "patch":
        sec = "sky/patch"
        xsize = configs.getn(sec+"/xsize")
        ysize = configs.getn(sec+"/ysize")
        xcenter = configs.getn(sec+"/xcenter")
        ycenter = configs.getn(sec+"/ycenter")
        pixelsize = configs.getn(sec+"/pixelsize")
        return SkyPatch(size=(xsize, ysize), pixelsize=pixelsize,
                        center=(xcenter, ycenter), **kwargs)
    elif skytype == "healpix":
        sec = "sky/healpix"
        nside = configs.getn(sec+"/nside")
        return SkyHealpix(nside=nside, **kwargs)
    else:
        raise ValueError("unknown sky type: %s" % skytype)
