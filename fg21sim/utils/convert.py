# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Utilities for conversion among common astronomical quantities.
"""

import astropy.units as au


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
