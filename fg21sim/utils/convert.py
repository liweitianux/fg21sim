# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Utilities for conversion among common astronomical quantities.
"""

import numpy as np
import astropy.units as au
import numba


def Fnu_to_Tb(Fnu, omega, freq):
    """Convert flux density to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    Parameters
    ----------
    Fnu : `~astropy.units.Quantity`
        Input flux density, e.g., `1.0*au.Jy`
    omega : `~astropy.units.Quantity`
        Source angular size/area, e.g., `1.0*au.sr`
    freq : `~astropy.units.Quantity`
        Frequency where the flux density measured, e.g., `1.0*au.MHz`

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
    """Convert surface brightness to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    Parameters
    ----------
    Sb : `~astropy.units.Quantity`
        Input surface brightness, e.g., `1.0*au.Jy/au.sr`
    freq : `~astropy.units.Quantity`
        Frequency where the flux density measured, e.g., `1.0*au.MHz`

    Returns
    -------
    Tb : `~astropy.units.Quantity`
        Brightness temperature, with default unit `au.K`
    """
    omega = 1.0 * au.sr
    Fnu = Sb * omega
    return Fnu_to_Tb(Fnu, omega, freq)


@numba.jit(nopython=True)
def Sb_to_Tb_fast(Sb, freq):
    """Convert surface brightness to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    This function does the calculations explicitly, and does NOT rely
    on the `astropy.units`, therefore it is faster.  However, the input
    parameters must be in right units.

        Tb = Sb * c^2 / (2 * k_B * nu^2)

    where `SB` is the surface brightness density measured at a certain
    frequency (unit: [ Jy/rad^2 ] = [ erg/s/cm^2/Hz/rad^2 ]).

    Parameters
    ----------
    Sb : float
        Input surface brightness, unit [ Jy/deg^2 ]
    freq : float
        Frequency where the flux density measured, unit [ MHz ]

    Returns
    -------
    Tb : float
        Calculated brightness temperature, unit [ K ]
    """
    # NOTE: `radian` is dimensionless
    rad2_to_deg2 = np.rad2deg(1.0) * np.rad2deg(1.0)
    Sb_rad2 = Sb * rad2_to_deg2  # unit: [ Jy/rad^2 ] -> [ Jy ]
    c = 29979245800.0  # speed of light, [ cm/s ]
    k_B = 1.3806488e-16  # Boltzmann constant, [ erg/K ]
    coef = 1e-35  # take care the unit conversions
    Tb = coef * (Sb_rad2 * c*c) / (2 * k_B * freq*freq)  # unit: [ K ]
    return Tb


@numba.jit(nopython=True)
def Fnu_to_Tb_fast(Fnu, omega, freq):
    """Convert flux density to brightness temperature, using the
    Rayleigh-Jeans law, in the Rayleigh-Jeans limit.

    This function does the calculations explicitly, and does NOT rely
    on the `astropy.units`, therefore it is faster.  However, the input
    parameters must be in right units.

    Parameters
    ----------
    Fnu : float
        Input flux density, unit [ Jy ]
    omega : float
        Source angular size/area, unit [ deg^2 ]
    freq : float
        Frequency where the flux density measured, unit [ MHz ]

    Returns
    -------
    Tb : float
        Calculated brightness temperature, unit [ K ]
    """
    Sb = Fnu / omega
    return Sb_to_Tb_fast(Sb, freq)
