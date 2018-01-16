# Copyright (c) 2017-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Functions to help simulate galaxy cluster diffuse emissions.

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

.. [fujita2003]
   Fujita et al. 2003, ApJ, 584, 190;
   http://adsabs.harvard.edu/abs/2003ApJ...584..190F

.. [lokas2001]
   Lokas & Mamon 2001, MNRAS, 321, 155
   http://adsabs.harvard.edu/abs/2001MNRAS.321..155L

.. [murgia2009]
   Murgia et al. 2009, A&A, 499, 679
   http://adsabs.harvard.edu/abs/2009A%26A...499..679M

.. [vazza2011]
   Vazza et al. 2011, A&A, 529, A17
   http://adsabs.harvard.edu/abs/2011A%26A...529A..17V

.. [zandanel2014]
   Zandanel, Pfrommer & Prada 2014, MNRAS, 438, 124
   http://adsabs.harvard.edu/abs/2014MNRAS.438..124Z

.. [zhuravleva2014]
   Zhuravleva et al. 2014, Nature, 515, 85;
   http://adsabs.harvard.edu/abs/2014Natur.515...85Z
"""

import logging

import numpy as np
from scipy import integrate

from ...share import CONFIGS, COSMO
from ...utils.units import (Units as AU,
                            Constants as AC,
                            UnitConversions as AUC)
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


def radius_halo(mass, z=0.0, configs=CONFIGS):
    """
    Estimate the radius of (giant) radio halo.

    NOTE
    ----
    The halo radius is estimated to be the same as the turbulence
    injection scale, i.e.:
        R_halo ≅ L ≅ R_vir / 3
    where R_vir the virial radius of the merged (observed) cluster.

    Reference: [vazza2011],Sec.(3.6)

    Parameters
    ----------
    mass : float, `~numpy.ndarray`
        Cluster virial mass.
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
    # Turbulence injection scale factor
    key = "extragalactic/halos/f_lturb"
    f_lturb = configs.getn(key)
    R_halo = f_lturb * radius_virial(mass=mass, z=z)  # [kpc]
    return R_halo


def kT_virial(mass, z=0.0, radius=None):
    """
    Calculate the virial temperature of a cluster.

    Parameters
    ----------
    mass : float
        The virial mass of the cluster.
        Unit: [Msun]
    z : float, optional
        The redshift of the cluster.
    radius : float, optional
        The virial radius of the cluster.
        If no provided, then invoke the above ``radius_virial()``
        function to calculate it.
        Unit: [kpc]

    Returns
    -------
    kT : float
       The virial temperature of the cluster.
       Unit: [keV]

    Reference: Ref.[fujita2003],Eq.(48)
    """
    if radius is None:
        radius = radius_virial(mass=mass, z=z)  # [kpc]
    kT = AC.mu*AC.u * AC.G * mass*AUC.Msun2g / (2*radius*AUC.kpc2cm)  # [erg]
    kT *= AUC.erg2keV  # [keV]
    return kT


def kT_cluster(mass, z=0.0, radius=None, configs=CONFIGS):
    """
    Calculate the temperature of a cluster ICM.

    NOTE
    ----
    When a cluster forms, there are accretion shocks forms around
    the cluster (near the virial radius) which can heat the gas,
    therefore the ICM has a higher temperature than the virial
    temperature, which can be estimated as:
        kT_icm ~ kT_vir + 1.5 * kT_out
    where kT_out the temperature of the outer gas surround the cluster,
    which may be ~0.5-1.0 keV.

    Reference: Ref.[fujita2003],Eq.(49)

    Returns
    -------
    kT_icm : float
       The temperature of the cluster ICM.
       Unit: [keV]
    """
    key = "extragalactic/clusters/kT_out"
    kT_out = configs.getn(key)
    kT_vir = kT_virial(mass=mass, z=z, radius=radius)
    kT_icm = kT_vir + 1.5*kT_out
    return kT_icm


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


def density_energy_thermal(mass, z=0.0, configs=CONFIGS):
    """
    Calculate the thermal energy density of the ICM.

    Returns
    -------
    e_th : float
        Energy density of the ICM
        Unit: [erg/cm^3]
    """
    n_th = density_number_thermal(mass=mass, z=z)  # [cm^-3]
    kT = kT_cluster(mass, z, configs=configs) * AUC.keV2erg  # [erg]
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


def magnetic_field(mass, z=0.0, configs=CONFIGS):
    """
    Calculate the mean magnetic field strength within the ICM, which is
    also assumed to be uniform, according to the assumed fraction of the
    the magnetic field energy density w.r.t. the ICM thermal energy density.

    NOTE
    ----
    Magnetic field energy density: u_B = B^2 / (8π),
    where "B" in units of [G], then "u_B" has unit of [erg/cm^3].

    Returns
    -------
    B : float
        The mean magnetic field strength within the ICM.
        Unit: [uG]
    """
    key = "extragalactic/clusters/eta_b"
    eta_b = configs.getn(key)
    e_th = density_energy_thermal(mass=mass, z=z, configs=configs)
    B = np.sqrt(8*np.pi * eta_b * e_th) * 1e6  # [G] -> [uG]
    return B


def speed_sound(kT):
    """
    The adiabatic sound speed in cluster ICM.

    Parameters
    ----------
    kT : float
        The cluster ICM temperature
        Unit: [keV]

    Returns
    -------
    cs : float
        The speed of sound in cluster ICM.
        Unit: [km/s]

    Reference: Ref.[zhuravleva2014],Appendix(Methods)
    """
    gamma = AC.gamma  # gas adiabatic index
    cs = np.sqrt(gamma * kT*AUC.keV2erg / (AC.mu * AC.u))  # [cm/s]
    return cs * AUC.cm2km  # [km/s]


def velocity_virial(mass, z=0.0):
    """
    Calculate the virial velocity, i.e., free-fall velocity.

    Unit: [km/s]
    """
    R_vir = radius_virial(mass, z) * AUC.kpc2cm  # [cm]
    vv = np.sqrt(AC.G * mass*AUC.Msun2g / R_vir)  # [cm/s]
    vv /= AUC.km2cm  # [km/s]
    return vv


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
    eta_v = 4 * (1 + M_main/M_sub) ** 0.333333
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
    uconv = AUC.kpc2km * AUC.s2Gyr  # [s kpc/km] => [Gyr]
    time = uconv * R_vir / vi  # [Gyr]
    return time


def time_turbulence(M_main, M_sub, z=0.0, configs=CONFIGS):
    """
    The duration that the compressive turbulence persists, which is
    estimated as:
        τ_turb ≅ 2*d / v_impact,
    where d ≅ L ≅ R_vir / 3,
    and L is also the turbulence injection scale.
    During this period, the merger-induced turbulence is regarded
    to accelerate the relativistic electrons effectively.

    Unit: [Gyr]
    """
    # Turbulence injection scale factor
    key = "extragalactic/halos/f_lturb"
    f_lturb = configs.getn(key)
    R_vir = radius_virial(M_main, z)  # [kpc]
    distance = 2*R_vir * f_lturb
    vi = velocity_impact(M_main, M_sub, z)  # [km/s]
    uconv = AUC.kpc2km * AUC.s2Gyr  # [s kpc/km] => [Gyr]
    time = uconv * distance / vi  # [Gyr]
    return time


def draw_halo(radius, nr=2.0, felong=None, rotation=None):
    """
    Draw the template image of one halo, which is used to simulate
    the image at requested frequencies by adjusting the brightness
    values.

    NOTE
    ----
    The exponential radial profile is adopted for radio halos:
        I(r) = I0 * exp(-r/re)
    with the e-folding radius ``re ~ R_halo / 3``.

    Reference: Ref.[murgia2009],Eq.(1)

    Parameters
    ----------
    radius : float
        The halo radius in number of pixels.
    nr : float, optional
        The times of ``radius`` to determine the size of the template
        image.
        Default: 2.0 (corresponding to 3*2=6 re)
    felong : float, optional
        The elongated fraction of the elliptical halo, which is
        defined as the ratio of semi-minor axis to the semi-major axis.
        Default: ``None`` (i.e., circular halo)
    rotation : float, optional
        The rotation angle of the elliptical halo.
        Unit: [deg]
        Default: ``None`` (i.e., no rotation)

    Returns
    -------
    image : 2D `~numpy.ndarray`
        2D array of the drawn halo template image.
        The image is normalized to have *mean* value of 1.
    """
    # Make halo radial brightness profile
    re = radius / 3.0  # e-folding radius
    # NOTE: Use ``ceil()`` here to make sure ``rprofile`` has length >= 2,
    #       therefore the interpolation in ``circle()`` runs well.
    rmax = int(np.ceil(radius*nr))
    r = np.arange(rmax+1)
    rprofile = np.exp(-r/re)

    image = circle(rprofile=rprofile)
    if felong:
        image = circle2ellipse(image, bfraction=felong, rotation=rotation)

    # Normalized to have *mean* value of 1
    image /= image.mean()
    return image


def fmass_nfw(x, c=5.0):
    """
    The normalized total mass profile by integrating from the NFW
    density profile.

    Parameters
    ----------
    x : float
        x = R/R_vir, fractional virial radius
    c : float
        Concentration parameter
        Default: 5.0 (for clusters)

    Returns
    -------
    fmass : float
        The normalized total mass w.r.t. the virial mass, i.e.,
        fmass = M(<x*R_vir) / M_vir

    Reference: [lokas2001],Eq.(2,4,5,8)
    """
    gc = 1.0 / (np.log(1+c) - c/(1+c))
    fmass = gc * (np.log(1+c*x) - c*x / (1+c*x))
    return fmass
