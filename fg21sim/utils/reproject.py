# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
FITS WCS (world coordinate system) reprojection utilities.

zea2healpix:
    Reproject the maps in ZEA (zenithal/azimuthal equal area) projection
    to Galactic frame and organize in HEALPix format.
"""

import logging

import numpy as np
from scipy.ndimage import map_coordinates
import astropy.units as au
from astropy.coordinates import Galactic, UnitSphericalRepresentation
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.io import fits
import healpy as hp

from .healpix import _make_healpix_header


logger = logging.getLogger(__name__)


def _convert_wcs(lon_in, lat_in, frame_in, frame_out):
    """Convert (longitude, latitude) coordinates from the input frame to
    the specified output frame.

    Parameters
    ----------
    lon_in : 1D `~numpy.ndarray`
        The longitude to convert, unit degree, [0, 360)
    lat_in : 1D `~numpy.ndarray`
        The latitude to convert, unit degree, [-90, 90]
    frame_in, frame_out : tuple or `~astropy.wcs.WCS`
        The input and output frames, which can be passed either as a tuple of
        ``(frame, lon_unit, lat_unit)`` or as a `~astropy.wcs.WCS` instance.

    Returns
    -------
    lon_out, lat_out : 1D `~numpy.ndarray`
        Output longitude and latitude in the output frame

    References
    ----------
    [1] reproject - wcs_utils.convert_world_coordinates()
        https://github.com/astrofrog/reproject
    """
    if isinstance(frame_in, WCS):
        coordframe_in = wcs_to_celestial_frame(frame_in)
        lon_in_unit = au.Unit(frame_in.wcs.cunit[0])
        lat_in_unit = au.Unit(frame_in.wcs.cunit[1])
    else:
        coordframe_in, lon_in_unit, lat_in_unit = frame_in
    #
    if isinstance(frame_out, WCS):
        coordframe_out = wcs_to_celestial_frame(frame_out)
        lon_out_unit = au.Unit(frame_out.wcs.cunit[0])
        lat_out_unit = au.Unit(frame_out.wcs.cunit[1])
    else:
        coordframe_out, lon_out_unit, lat_out_unit = frame_out
    #
    logger.info("Convert coordinates from {0} to {1}".format(coordframe_in,
                                                             coordframe_out))
    logger.info("Input coordinates units: "
                "{0} (longitude), {1} (latitude)".format(lon_in_unit,
                                                         lat_in_unit))
    logger.info("Output coordinates units: "
                "{0} (longitude), {1} (latitude)".format(lon_out_unit,
                                                         lat_out_unit))
    #
    data = UnitSphericalRepresentation(lon_in*lon_in_unit, lat_in*lat_in_unit)
    coords_in = coordframe_in.realize_frame(data)
    coords_out = coords_in.transform_to(coordframe_out)
    data_out = coords_out.represent_as("unitspherical")
    lon_out = data_out.lon.to(lon_out_unit).value
    lat_out = data_out.lat.to(lon_out_unit).value
    return lon_out, lat_out


def _image_to_healpix(image, wcs, nside, order=1, hemisphere=None):
    """Convert image in a normal WCS projection to HEALPix data of *RING*
    ordering and *Galactic* coordinate system.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Input image array to be reprojected into HEALPix format.
    wcs : `~astropy.wcs.WCS`
        The WCS of the input image.
    order : int, optional
        The order of the spline interpolation, valid range: 0-5
    hemisphere : str, optional
        Specify the hemisphere on which the pixels to be reprojected.
        Valid values: `"N"/"NORTH"/"NORTHERN"` (northern),
        `"S"/"SOUTH"/"SOUTHERN"` (southern), or `None`.
        If None, then no pixel filtering applied.
        Note: northern hemisphere includes the equator, while southern not.

    Returns
    -------
    hpdata : 1D `~numpy.ndarray`
        Projected HEALPix data array (1D) of length 12*nside*nside.
        The invalid pixels are filled with value NaN (`np.nan`).

    NOTE
    ----
    Since the HEALPix map is full-sky, however, the input image may contains
    only part of the full sky (e.g., one of the ZEA-projected image contains
    only either northern or southern Galactic hemisphere), so the argument
    `hemisphere` should be specified if applicable.
    Otherwise, the WCS (of input image) can NOT correctly convert the
    requested sky coordinates to the pixels in the input image.
    If the requested coordinates beyond the available scope of the input
    image, then the converted pixel positions may be negative or even WRONG.

    References
    ----------
    [1] reproject - healpix.core.image_to_healpix()
        https://github.com/astrofrog/reproject
    """
    if (hemisphere is not None) and (
            hemisphere.upper() not in ["N", "NORTH", "NORTHERN",
                                       "S", "SOUTH", "SOUTHERN"]):
        raise ValueError("invalid hemisphere: {0}".format(hemisphere))
    #
    npix = hp.nside2npix(nside)
    hpidx = np.arange(npix).astype(np.int)
    logger.info("Output HEALPix: Nside={0}, Npixel={1}".format(nside, npix))
    # Calculate the longitude and latitude in frame of output HEALPix
    logger.info("Calculate the longitudes and latitudes on the HEALPix grid")
    theta, phi = hp.pix2ang(nside, hpidx, nest=False)
    lon_hp = np.degrees(phi)
    lat_hp = 90.0 - np.degrees(theta)
    # Convert between the celestial coordinate systems
    coordsys_hp = Galactic()
    logger.info("Output HEALPix uses frame: {0}".format(coordsys_hp))
    frame_hp = (coordsys_hp, au.deg, au.deg)
    lon_in, lat_in = _convert_wcs(lon_hp, lat_hp, frame_hp, wcs)
    # Filter the pixels on the specified hemisphere
    if hemisphere is None:
        mask = np.ones(lat_in.shape).astype(np.bool)
        logger.info("NO hemisphere constraint specified")
    elif hemisphere.upper() in ["N", "NORTH", "NORTHERN"]:
        # northern hemisphere (include the equator)
        mask = lat_in >= 0.0
        logger.info("Only process the NORTHERN hemisphere (include EQUATOR")
    else:
        # southern hemisphere
        mask = lat_in < 0.0
        logger.info("Only process the SOUTHERN hemisphere (EXCLUDE equator")
    lon_in = lon_in[mask]
    lat_in = lat_in[mask]
    # Look up pixels in the input coordinate system
    # XXX/NOTE: note the order of returns: (Y, X)
    yi, xi = wcs.wcs_world2pix(lon_in, lat_in, 0)
    # Interpolate to obtain the HEALPix data from the input image
    logger.info("Calculate the HEALPix data by interpolating on input image")
    logger.info("Interpolation order: {0}".format(order))
    data = map_coordinates(image, [xi, yi], order=order,
                           mode="constant", cval=np.nan)
    # Make the HEALPix array with above hemisphere mask considered
    hpdata = np.zeros(shape=npix, dtype=data.dtype)
    hpdata[mask] = data
    hpdata[~mask] = np.nan
    return hpdata


def zea2healpix(img1, img2, nside, order=1, inpaint=False,
                append_history=None, append_comment=None):
    """Reproject the maps in ZEA (zenithal/azimuthal equal area) projection
    to Galactic frame and organize in HEALPix format.

    Parameters
    ----------
    img1, img2 : str or `~astropy.io.fits.PrimaryHDU`
        Two input ZEA-projected FITS files
    nside : int
        Nside for the output HEALPix data
    order : int, optional
        Interpolation order, valid range: 0-5
    inpaint : bool, optional
        Whether to inpaint the missing pixels
    append_history : list[str], optional
        Append the provided history to the output FITS header
    append_comment : list[str], optional
        Append the provided comment to the output FITS header

    Returns
    -------
    hp_data : 1D `~numpy.ndarray`
        Reprojected HEALPix data
    hp_header : `~astropy.io.fits.Header`
        FITS header for the reprojected HEALPix data
    hp_mask : 1D `~numpy.ndarray`
        Array of same shape as the above `hp_data` indicating the status of
        each pixel of the output array.
        Values of "0" indicate the missing pixels (i.e., there is no
        transformation to the input images); values of "1" indicate the output
        pixel maps to one and only one of the input images; values of "2"
        indicate the duplicate/overlapping pixels that map to both of the two
        input images.

    NOTE
    ----
    - One ZEA-projected FITS file only contains either the northern Galactic
      hemisphere (LAM_NSGP=1), or southern Galactic hemisphere (LAM_NSGP=-1).
      Thus two ZEA-projected FITS files should both be provided to get the
      full-sky map.
    - The two reprojected HEALPix data are simply added to compose the
      full-sky HEALPix map.  Duplicate/overlapping pixels are warned.
    - The combined full-sky HEALPix map may still have some missing pixels,
      which is also warned.
      TODO: inpaint the missing pixels by interpolation
    """
    if isinstance(img1, str):
        img1 = fits.open(img1)[0]
    if isinstance(img2, str):
        img2 = fits.open(img2)[0]
    zea_img1, zea_hdr1 = img1.data, img1.header
    zea_img2, zea_hdr2 = img2.data, img2.header
    ZEA_NSGP = {1: "Northern", -1: "Southern"}
    zea_hemisphere1 = ZEA_NSGP.get(zea_hdr1["LAM_NSGP"])
    zea_hemisphere2 = ZEA_NSGP.get(zea_hdr2["LAM_NSGP"])
    logger.info("Read ZEA image1: {0}, shape {1}".format(
        zea_hemisphere1, zea_img1.shape))
    logger.info("Read ZEA image2: {0}, shape {1}".format(
        zea_hemisphere2, zea_img2.shape))
    zea_wcs1 = WCS(zea_hdr1)
    zea_wcs2 = WCS(zea_hdr2)
    logger.info("Reproject ZEA images to HEALPix ...")
    hp_data1 = _image_to_healpix(zea_img1, zea_wcs1, nside=nside, order=order,
                                 hemisphere=zea_hemisphere1.upper())
    hp_data2 = _image_to_healpix(zea_img2, zea_wcs2, nside=nside, order=order,
                                 hemisphere=zea_hemisphere2.upper())
    # Merge the two HEALPix data
    hp_mask = ((~np.isnan(hp_data1)).astype(np.int) +
               (~np.isnan(hp_data2)).astype(np.int))
    hp_data1[np.isnan(hp_data1)] = 0.0
    hp_data2[np.isnan(hp_data2)] = 0.0
    hp_data = hp_data1 + hp_data2
    logger.info("Done reprojection and merge two hemispheres")
    # Duplicate pixels and missing pixels
    pix_dup = (hp_mask == 2)
    if pix_dup.sum() > 0:
        logger.warning("Reprojected HEALPix data has %d duplicate pixel(s)" %
                       pix_dup.sum())
        hp_data[pix_dup] /= 2.0
        logger.warning("Averaged the duplicate pixel(s)")
    pix_missing = (hp_mask == 0)
    if pix_missing.sum() > 0:
        logger.warning("Reprojected HEALPix data has %d missing pixel(s)" %
                       pix_missing.sum())
        # XXX/TODO: inpaint
    # HEALPix FITS header
    header = zea_hdr1.copy(strip=True)
    keys = ["CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "LONPOLE", "LAM_NSGP", "LAM_SCAL"]
    for k in keys:
        if k in header:
            del header[k]
    hp_header = _make_healpix_header(header, nside=nside,
                                     append_history=append_history,
                                     append_comment=append_comment)
    logger.info("Made HEALPix FITS header")
    return (hp_data, hp_header, hp_mask)
