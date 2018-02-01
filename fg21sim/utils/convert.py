# Copyright (c) 2016-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Utilities for conversion among common astronomical quantities.
"""

import numpy as np

from .units import (UnitConversions as AUC, Constants as AC)
from ..share import COSMO


def Sb_to_Tb(Sb, freq):
    """
    Convert surface brightness to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

        Tb = Sb * c^2 / (2 * k_B * nu^2)

    where `Sb` is the surface brightness density measured at a certain
    frequency, in units of [Jy/arcsec^2].

    1 [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]

    NOTE
    ----
    It is very easy to use ``astropy.units`` for the conversion:
        equiv = au.brightness_temperature(omega, freq)
        Tb = Fnu.to(au.K, equivalencies=equiv)

    WARNING:
    Using `astropy.units` for the conversion may be much slower.
    This can be used as a cross check for the calculation.

    Parameters
    ----------
    Sb : float
        Input surface brightness
        Unit: [Jy/arcsec^2]
    freq : float
        Frequency where the flux density measured
        Unit: [MHz]

    Returns
    -------
    Tb : float
        Calculated brightness temperature
        Unit: [K]

    References
    ----------
    - Brightness and Flux
      http://www.cv.nrao.edu/course/astr534/Brightness.html
    - Wikipedia: Brightness Temperature
      https://en.wikipedia.org/wiki/Brightness_temperature
    - NJIT: Physics 728: Introduction to Radio Astronomy: Lecture #1
      https://web.njit.edu/~gary/728/Lecture1.html
    - Astropy: Equivalencies: Brightness Temperature / Flux Density
      http://docs.astropy.org/en/stable/units/equivalencies.html
    """
    # NOTE: [rad] & [sr] are dimensionless
    arcsec2 = AUC.arcsec2rad ** 2  # [sr]
    Sb /= arcsec2  # [Jy/arcsec^2] -> [Jy/sr]
    coef = 1e-35  # unit conversion coefficient
    Tb = coef * (Sb * AC.c**2) / (2*AC.k_B * freq**2)  # [K]
    return Tb


def Fnu_to_Tb(Fnu, omega, freq):
    """
    Convert flux density to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    Avoid using `astropy.units` to optimize the speed.

    Parameters
    ----------
    Fnu : float
        Input flux density
        Unit: [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]
    omega : float
        Source angular size/area
        Unit: [arcsec^2]
    freq : float
        Frequency where the flux density measured
        Unit: [MHz]

    Returns
    -------
    Tb : float
        Calculated brightness temperature
        Unit: [K]
    """
    Sb = Fnu / omega  # [Jy/arcsec^2]
    return Sb_to_Tb(Sb, freq)


def JyPerPix_to_K(freq, pixelsize):
    """
    The factor that converts [Jy/pixel] to [K] (brightness temperature).

    Parameters
    ----------
    freq : float
        The frequency where the flux density measured.
        Unit: [Jy]
    pixelsize : float
        The pixel size.
        Unit: [arcsec]
    """
    factor = Fnu_to_Tb(Fnu=1.0, omega=pixelsize**2, freq=freq)
    return factor


def flux_to_power(flux, z, index=1.0):
    """
    Calculate the (spectral) power from the measured flux density at
    the same frequency with K-correction taking into account, if the
    spectral index is given.

    Parameters
    ----------
    flux : float
        The measured flux density at a frequency.
        Unit: [mJy]
    z : float
        The redshift to the source.
    index : float, optional
        The spectral index (α) of the source flux: S(ν) ∝ ν^(-α)
        If given, the K-correction is taken into account.

    Returns
    -------
    power : float
        The calculated (spectral) power at the same frequency as
        the flux density measured.
        Unit: [W/Hz]
    """
    flux *= 1e-29  # [mJy] -> [W/Hz/m^2]
    DL = COSMO.DL(z) * AUC.Mpc2m  # [m]
    power = 4*np.pi * DL**2 * (1+z)**(index-1) * flux  # [W/Hz]
    return power
