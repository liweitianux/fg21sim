# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Helper functions

References
----------
.. [cassano2007]
   Cassano et al. 2007, MNRAS, 378, 1565;
   http://adsabs.harvard.edu/abs/2007MNRAS.378.1565C
.. [arnaud2005]
   Arnaud, Pointecouteau & Pratt 2005, A&A, 441, 893;
   http://adsabs.harvard.edu/abs/2005A%26A...441..893
"""


import numpy as np

from ...utils import cosmo
from ...utils.units import (Constants as AC,
                            UnitConversions as AUC)


def radius_virial(mass, z=0.0):
    """
    Calculate the virial radius of a cluster at a given redshift.

    Parameters
    ----------
    mass : float
        Total (virial) mass of the cluster
        Unit: [Msun]
    z : float, optional
        Redshift
        Default: 0.0 (i.e., present day)

    Returns
    -------
    R_vir : float
        Virial radius of the cluster
        Unit: [kpc]
    """
    Dc = cosmo.overdensity_virial(z)
    rho = cosmo.rho_crit(z)  # [g/cm^3]
    R_vir = (3*mass*AUC.Msun2g / (4*np.pi * Dc * rho))**(1/3)  # [cm]
    R_vir *= AUC.cm2kpc  # [kpc]
    return R_vir


def radius_halo(mass, z=0.0):
    """
    Calculate the radius of (giant) radio halo for a cluster.

    The halo radius is derived from the virial radius using the scaling
    relation in [cassano2007]_.

    Parameters
    ----------
    mass : float
        Total (virial) mass of the cluster
        Unit: [Msun]
    z : float, optional
        Redshift
        Default: 0.0 (i.e., present day)

    Returns
    -------
    R_halo : float
        Radius of the (expected) giant radio halo
        Unit: [kpc]

    References
    ----------
    Ref.[cassano2007],Fig.(11)
    """
    slope = 2.63 + np.random.normal(scale=0.5)
    intercept = 2.3 + np.random.normal(scale=0.05)
    R_vir = radius_virial(mass=mass, z=z)  # [kpc]
    R_halo = 10 ** (slope * np.log10(R_vir) + intercept)
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
    A = 5.34 + np.random.normal(scale=0.22)
    alpha = 1.72 + np.random.normal(scale=0.10)
    Ez = cosmo.E(z)
    kT = 5.0 * (mass * Ez / A) ** (1/alpha)
    return kT


def density_number_thermal(mass, z=0.0):
    """
    Calculate the number density of the ICM thermal plasma.

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
    N = mass * AUC.Msun2g * cosmo.baryon_fraction / (AC.mu * AC.u)
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
        Energy density of the ICM (unit: erg/cm^3)
    """
    n_th = density_number_thermal(mass, z)  # [cm^-3]
    kT = mass_to_kT(mass, z) * AUC.keV2erg  # [erg]
    e_th = (3.0/2) * kT * n_th
    return e_th
