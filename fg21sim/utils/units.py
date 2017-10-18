# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Commonly used units and their conversions relations, as well as constants.

Astropy's units system is very powerful, but also very slow,
and may even be the speed bottleneck of the program.

This module provides commonly used units conversions by holding
them directly in a class, thus avoid repeated/unnecessary calculations.
"""

import astropy.units as au
import astropy.constants as ac


class Units:
    """
    Commonly used units, especially in the CGS unit system.
    """
    # Unit for electron momentum (p), thus its value is the Lorentz factor
    # Unit: [g cm / s]
    mec = ac.m_e.cgs.value * ac.c.cgs.value
    # Energy of a still electron
    # Unit: [erg]
    mec2 = (ac.m_e * ac.c**2).to(au.erg).value


class UnitConversions:
    """
    Commonly used units conversion relations.

    Hold the conversion relations directly to avoid repeated/unnecessary
    calculations.
    """
    # Mass
    Msun2g = au.solMass.to(au.g)
    g2Msun = 1.0 / Msun2g
    # Time
    Gyr2s = au.Gyr.to(au.s)
    s2Gyr = 1.0 / Gyr2s
    # Length
    kpc2m = au.kpc.to(au.m)
    m2kpc = 1.0 / kpc2m
    Mpc2m = au.Mpc.to(au.m)
    m2Mpc = 1.0 / Mpc2m
    kpc2cm = au.kpc.to(au.cm)
    cm2kpc = 1.0 / kpc2cm
    Mpc2cm = au.Mpc.to(au.cm)
    cm2Mpc = 1.0 / Mpc2cm
    Mpc2km = au.Mpc.to(au.km)
    km2Mpc = 1.0 / Mpc2km
    kpc2km = au.kpc.to(au.km)
    km2kpc = 1.0 / kpc2km
    km2cm = au.km.to(au.cm)
    cm2km = 1.0 / km2cm
    # Energy
    keV2erg = au.keV.to(au.erg)
    erg2keV = 1.0 / keV2erg
    # Angle
    rad2deg = au.rad.to(au.deg)
    deg2rad = 1.0 / rad2deg
    rad2arcsec = au.rad.to(au.arcsec)
    arcsec2rad = 1.0 / rad2arcsec
    rad2arcmin = au.rad.to(au.arcmin)
    arcmin2rad = 1.0 / rad2arcmin
    deg2arcmin = au.deg.to(au.arcmin)
    arcmin2deg = 1.0 / deg2arcmin
    deg2arcsec = au.deg.to(au.arcsec)
    arcsec2deg = 1.0 / deg2arcsec
    arcmin2arcsec = au.arcmin.to(au.arcsec)
    arcsec2arcmin = 1.0 / arcmin2arcsec
    # Temperature
    eV2K = au.eV.to(ac.k_B*au.K)
    K2eV = 1.0 / eV2K
    keV2K = au.keV.to(ac.k_B*au.K)
    K2keV = 1.0 / keV2K


class Constants:
    """
    Commonly used constants, especially in the CGS unit system.

    Astropy's constants are stored in SI units by default.
    When request a constant in CGS unit system, additional (and slow)
    conversions required.
    """
    # Speed of light
    c = ac.c.cgs.value  # [cm/s]
    # Atomic mass unit (i.e., a.m.u.)
    u = ac.u.cgs.value  # [g]
    # Gravitational constant
    G = ac.G.cgs.value  # [cm^3/g/s^2]
    # Electron charge
    e = ac.e.gauss.value  # [Fr] = [esu]
    # Boltzmann constant
    k_B = ac.k_B.cgs.value  # [erg/K]

    # Mean molecular weight
    # Ref.: Ettori et al, 2013, Space Science Review, 177, 119-154, Eq.(6)
    mu = 0.6
