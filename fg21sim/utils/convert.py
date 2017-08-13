# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Utilities for conversion among common astronomical quantities.
"""

import numpy as np
import astropy.units as au
import numba


def Fnu_to_Tb(Fnu, omega, freq):
    """
    Convert flux density to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    Parameters
    ----------
    Fnu : `~astropy.units.Quantity`
        Input flux density, e.g., `1.0*au.Jy`
    omega : `~astropy.units.Quantity`
        Source angular size/area, e.g., `100*au.arcsec**2`
    freq : `~astropy.units.Quantity`
        Frequency where the flux density measured, e.g., `120*au.MHz`

    Returns
    -------
    Tb : `~astropy.units.Quantity`
        Brightness temperature, with default unit `au.K`

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
    equiv = au.brightness_temperature(omega, freq)
    Tb = Fnu.to(au.K, equivalencies=equiv)
    return Tb


def Sb_to_Tb(Sb, freq):
    """
    Convert surface brightness to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    Parameters
    ----------
    Sb : `~astropy.units.Quantity`
        Input surface brightness, e.g., `1.0*(au.Jy/au.arcsec**2)`
    freq : `~astropy.units.Quantity`
        Frequency where the flux density measured, e.g., `120*au.MHz`

    Returns
    -------
    Tb : `~astropy.units.Quantity`
        Brightness temperature, with default unit `au.K`
    """
    omega = 1.0 * au.arcsec**2
    Fnu = (Sb * omega).to(au.Jy)  # [Jy]
    return Fnu_to_Tb(Fnu, omega, freq)


@numba.jit(nopython=True)
def Sb_to_Tb_fast(Sb, freq):
    """
    Convert surface brightness to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    This function does the calculations explicitly, and does NOT rely
    on the `astropy.units`, therefore it is much faster.  However, the
    input parameters must be in right units.

        Tb = Sb * c^2 / (2 * k_B * nu^2)

    where `Sb` is the surface brightness density measured at a certain
    frequency, in units of [Jy/arcsec^2].

    1 [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]

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
    """
    # NOTE: [rad] & [sr] are dimensionless
    arcsec2 = (np.deg2rad(1) / 3600) ** 2  # [sr]
    c = 29979245800.0  # speed of light, [cm/s]
    k_B = 1.3806488e-16  # Boltzmann constant, [erg/K]
    coef = 1e-35  # take care the unit conversions
    Tb = coef * (Sb * c*c) / (2 * k_B * freq*freq * arcsec2)  # [K]
    return Tb


@numba.jit(nopython=True)
def Fnu_to_Tb_fast(Fnu, omega, freq):
    """
    Convert flux density to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    This function does NOT invoke the `astropy.units`, therefore it is
    much faster.

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
    return Sb_to_Tb_fast(Sb, freq)


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
    factor = Fnu_to_Tb_fast(Fnu=1.0, omega=pixelsize**2, freq=freq)
    return factor
