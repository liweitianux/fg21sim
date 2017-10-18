# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Helper functions

References
----------
.. [arnaud2005]
   Arnaud, Pointecouteau & Pratt 2005, A&A, 441, 893;
   http://adsabs.harvard.edu/abs/2005A%26A...441..893

.. [cassano2005]
   Cassano & Brunetti 2005, MNRAS, 357, 1313
   http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C

.. [cassano2007]
   Cassano et al. 2007, MNRAS, 378, 1565;
   http://adsabs.harvard.edu/abs/2007MNRAS.378.1565C

.. [cassano2012]
   Cassano et al. 2012, A&A, 548, A100
   http://adsabs.harvard.edu/abs/2012A%26A...548A.100C

.. [murgia2009]
   Murgia et al. 2009, A&A, 499, 679
   http://adsabs.harvard.edu/abs/2009A%26A...499..679M

.. [zandanel2014]
   Zandanel, Pfrommer & Prada 2014, MNRAS, 438, 124
   http://adsabs.harvard.edu/abs/2014MNRAS.438..124Z
"""

import logging

import numpy as np
from scipy import integrate

from ...share import CONFIGS, COSMO
from ...utils.units import (Units as AU,
                            Constants as AC,
                            UnitConversions as AUC)
from ...utils.convert import Fnu_to_Tb
from ...utils.draw import circle
from ...utils.transform import circle2ellipse


logger = logging.getLogger(__name__)


def radius_virial(mass, z=0.0):
    """
    Calculate the virial radius of a cluster at a given redshift.

    Parameters
    ----------
    mass : float, `~numpy.ndarray`
        Total (virial) mass of the cluster
        Unit: [Msun]
    z : float, `~numpy.ndarray`, optional
        Redshift
        Default: 0.0 (i.e., present day)

    Returns
    -------
    R_vir : float, `~numpy.ndarray`
        Virial radius of the cluster
        Unit: [kpc]
    """
    Dc = COSMO.overdensity_virial(z)
    rho = COSMO.rho_crit(z)  # [g/cm^3]
    R_vir = (3*mass*AUC.Msun2g / (4*np.pi * Dc * rho))**(1/3)  # [cm]
    R_vir *= AUC.cm2kpc  # [kpc]
    return R_vir


def radius_halo(M_main, M_sub, z=0.0):
    """
    Calculate the (predicted) radius of (giant) radio halo for a cluster.

    NOTE
    ----
    It can be intuitively assumed that a merger will generate turbulences
    within a region of size of the falling sub-cluster.  And this
    estimation can agree with the currently observed radio halos, which
    generally have a angular diameter size ~2-7 [arcmin].

    Parameters
    ----------
    M_main, M_sub : float, `~numpy.ndarray`
        Total (virial) masses of the main and sub clusters
        Unit: [Msun]
    z : float, `~numpy.ndarray`, optional
        Redshift
        Default: 0.0 (i.e., present day)

    Returns
    -------
    R_halo : float, `~numpy.ndarray`
        Radius of the (simulated/predicted) giant radio halo
        Unit: [kpc]
    """
    R_halo = radius_virial(mass=M_sub, z=z)  # [kpc]
    return R_halo


def mass_to_kT(mass, z=0.0):
    """
    Calculate the cluster ICM temperature from its mass using the
    mass-temperature scaling relation (its inversion used here)
    derived from observations.

    The following M-T scaling relation from Ref.[arnaud2005],Tab.2:
        M200 * E(z) = A200 * (kT / 5 keV)^α ,
    where:
        A200 = (5.34 +/- 0.22) [1e14 Msun]
        α = (1.72 +/- 0.10)
    Its inversion:
        kT = (5 keV) * [M200 * E(z) / A200]^(1/α).

    NOTE: M200 (i.e., Δ=200) is used to approximate the virial mass.

    Parameters
    ----------
    mass : float
        Total (virial) mass of the cluster.
        Unit: [Msun]
    z : float, optional
        Redshift of the cluster

    Returns
    -------
    kT : float
        The ICM mean temperature.
        Unit: [keV]
    """
    # A = (5.34 + np.random.normal(scale=0.22)) * 1e14  # [Msun]
    A = 5.34 * 1e14  # [Msun]
    # alpha = 1.72 + np.random.normal(scale=0.10)
    alpha = 1.72
    Ez = COSMO.E(z)
    kT = 5.0 * (mass * Ez / A) ** (1/alpha)
    return kT


def density_number_thermal(mass, z=0.0):
    """
    Calculate the number density of the ICM thermal plasma.

    NOTE
    ----
    This number density is independent of cluster (virial) mass,
    but (mostly) increases with redshifts.

    Parameters
    ----------
    mass : float
        Mass of the cluster
        Unit: [Msun]
    z : float, optional
        Redshift

    Returns
    -------
    n_th : float
        Number density of the ICM thermal plasma
        Unit: [cm^-3]
    """
    N = mass * AUC.Msun2g * COSMO.baryon_fraction / (AC.mu * AC.u)
    R_vir = radius_virial(mass, z) * AUC.kpc2cm  # [cm]
    volume = (4*np.pi / 3) * R_vir**3  # [cm^3]
    n_th = N / volume  # [cm^-3]
    return n_th


def density_energy_thermal(mass, z=0.0):
    """
    Calculate the thermal energy density of the ICM.

    Returns
    -------
    e_th : float
        Energy density of the ICM
        Unit: [erg cm^-3]
    """
    n_th = density_number_thermal(mass, z)  # [cm^-3]
    kT = mass_to_kT(mass, z) * AUC.keV2erg  # [erg]
    e_th = (3.0/2) * kT * n_th
    return e_th


def density_energy_electron(spectrum, gamma):
    """
    Calculate the energy density of relativistic electrons.

    Parameters
    ----------
    spectrum : 1D float `~numpy.ndarray`
        The number density of the electrons w.r.t. Lorentz factors
        Unit: [cm^-3]
    gamma : 1D float `~numpy.ndarray`
        The Lorentz factors of electrons

    Returns
    -------
    e_re : float
        The energy density of the relativistic electrons.
        Unit: [erg cm^-3]
    """
    e_re = integrate.trapz(spectrum*gamma*AU.mec2, gamma)
    return e_re


def velocity_impact(M_main, M_sub, z=0.0):
    """
    Estimate the relative impact velocity between the two merging
    clusters when they are at a distance of the virial radius.

    Parameters
    ----------
    M_main, M_sub : float
        Total (virial) masses of the main and sub clusters
        Unit: [Msun]
    z : float, optional
        Redshift

    Returns
    -------
    vi : float
        Relative impact velocity
        Unit: [km/s]

    References
    ----------
    Ref.[cassano2005],Eq.(9)
    """
    eta_v = 4 * (1 + M_main/M_sub) ** (1/3)
    R_vir = radius_virial(M_main, z) * AUC.kpc2cm  # [cm]
    vi = np.sqrt(2*AC.G * (1-1/eta_v) *
                 (M_main+M_sub)*AUC.Msun2g / R_vir)  # [cm/s]
    vi /= AUC.km2cm  # [km/s]
    return vi


def time_crossing(M_main, M_sub, z=0.0):
    """
    Estimate the crossing time of the sub cluster during a merger.

    NOTE: The crossing time is estimated to be τ ~ R_vir / v_impact.

    Parameters
    ----------
    M_main, M_sub : float
        Total (virial) masses of the main and sub clusters
        Unit: [Msun]
    z : float, optional
        Redshift

    Returns
    -------
    time : float
        Crossing time
        Unit: [Gyr]

    References
    ----------
    Ref.[cassano2005],Sec.(4.1)
    """
    R_vir = radius_virial(M_main, z)  # [kpc]
    vi = velocity_impact(M_main, M_sub, z)  # [km/s]
    # Unit conversion coefficient: [s kpc/km] => [Gyr]
    uconv = AUC.kpc2km * AUC.s2Gyr
    time = uconv * R_vir / vi  # [Gyr]
    return time


def magnetic_field(mass):
    """
    Calculate the mean magnetic field strength according to the
    scaling relation between magnetic field and cluster mass.

    Parameters
    ----------
    mass : float
        Cluster mass
        Unit: [Msun]

    Returns
    -------
    B : float
        The mean magnetic field strength
        Unit: [uG]

    References
    ----------
    Ref.[cassano2012],Eq.(1)
    """
    comp = "extragalactic/clusters"
    b_mean = CONFIGS.getn(comp+"/b_mean")
    b_index = CONFIGS.getn(comp+"/b_index")

    M_mean = 1.6e15  # [Msun]
    B = b_mean * (mass/M_mean) ** b_index
    return B


def calc_power(emissivity, volume):
    """
    Calculate the synchrotron power (i.e., power *emitted* per unit
    frequency) from emissivity, which assumed to be uniform within
    the volume.

    NOTE
    ----
    The calculated power (a.k.a. spectral luminosity) is in units of
    [W/Hz] which is common in radio astronomy, instead of [erg/s/Hz].
        1 [W] = 1e7 [erg/s]

    Parameters
    ----------
    emissivity : float, or 1D `~numpy.ndarray`
        The synchrotron emissivity at multiple frequencies.
        Unit: [erg/s/cm^3/Hz]
    volume : float
        The volume of the radio halo
        Unit: [kpc^3]

    Returns
    -------
    power : float, or 1D `~numpy.ndarray`
        The calculated synchrotron power w.r.t. each input emissivity.
        Unit: [W/Hz]
    """
    emissivity = np.asarray(emissivity)
    power = emissivity * (volume * AUC.kpc2cm**3)  # [erg/s/Hz]
    power *= 1e-7  # [erg/s/Hz] -> [W/Hz]
    return power


def calc_flux(power, z):
    """
    Calculate the synchrotron flux density (i.e., power *observed*
    per unit frequency) from radio power at a certain redshift (i.e.,
    distance).

    Parameters
    ----------
    power : float, or 1D `~numpy.ndarray`
        The synchrotron power at multiple frequencies.
        Unit: [W/Hz]

    Returns
    -------
    flux : float, or 1D `~numpy.ndarray`
        The calculated synchrotron flux w.r.t. each input power.
        Unit: [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]
    """
    DL = COSMO.DL(z) * AUC.Mpc2m  # [m]
    flux = 1e26 * power / (4*np.pi * DL*DL)  # [Jy]
    return flux


def calc_brightness_mean(flux, frequency, omega, pixelsize=None):
    """
    Calculate the mean surface brightness (power observed per unit
    frequency and per unit solid angle) expressed in *brightness
    temperature* at the specified frequencies from flux.

    NOTE
    ----
    If the solid angle that the object extends is smaller than the
    specified pixel area, then is is assumed to have size of 1 pixel.

    Parameters
    ----------
    flux : float, or 1D `~numpy.ndarray`
        The synchrotron flux densities at multiple frequencies.
        Unit: [Jy]
    frequency : float, or 1D `~numpy.ndarray`
        The frequencies where the above flux calculated.
        Unit: [MHz]
    omega : float
        The sky coverage (angular size) of the object.
        Unit: [arcsec^2]
    pixelsize : float, optional
        The pixel size of the output simulated sky image.
        Unit: [arcsec]

    Returns
    -------
    Tb : float, or 1D `~numpy.ndarray`
        The mean surface brightness at each frequency.
        Unit: [K] <-> [Jy/pixel]
    """
    if pixelsize and (omega < pixelsize**2):
        omega = pixelsize ** 2  # [arcsec^2]
        logger.warning("Object sky coverage < 1 pixel; force to be 1 pixel")

    Tb = [Fnu_to_Tb(Fnu, omega, freq)
          for Fnu, freq in zip(np.array(flux, ndmin=1),
                               np.array(frequency, ndmin=1))]
    if len(Tb) == 1:
        return Tb[0]
    else:
        return np.array(Tb)


def halo_rprofile(re, num_re=5, I0=1.0):
    """
    Generate the radial profile of a halo.

    NOTE
    ----
    The exponential radial profile is adopted for the radio halos:
        I(r) = I0 * exp(-r/re)
    with the e-folding radius ``re ~ R_halo / 3``.

    Parameters
    ----------
    re : float
        The e-folding radius in unit of pixels.
    num_re : float, optional
        The times of ``re`` to determine the maximum radius.
        Default: 5, i.e., rmax = 5 * re
    I0 : float
        The intensity/brightness at the center (i.e., r=0)
        Default: 1.0

    Returns
    -------
    rprofile : 1D `~numpy.ndarray`
        The values along the radial pixels (0, 1, 2, ...)

    References: Ref.[murgia2009],Eq.(1)
    """
    rmax = round(re * num_re)
    r = np.arange(rmax+1)
    rprofile = I0 * np.exp(-r/re)
    return rprofile


def draw_halo(rprofile, felong, rotation=0.0):
    """
    Draw the template image of one halo, which is used to simulate
    the image at requested frequencies by adjusting the brightness
    values.

    Parameters
    ----------
    rprofile : 1D `~numpy.ndarray`
        The values along the radial pixels (0, 1, 2, ...),
        e.g., calculated by the above ``halo_rprofile()``.
    felong : float
        The elongated fraction of the elliptical halo, which is
        defined as the ratio of semi-minor axis to the semi-major axis.
    rotation : float
        The rotation angle of the elliptical halo.
        Unit: [deg]

    Returns
    -------
    image : 2D `~numpy.ndarray`
        2D array of the drawn halo template image.
        The image is normalized to have *mean* value of 1.
    """
    image = circle(rprofile=rprofile)
    image = circle2ellipse(image, bfraction=felong, rotation=rotation)
    image /= image.mean()
    return image
