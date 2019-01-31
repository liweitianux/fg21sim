# Copyright (c) 2017-2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Functions to help simulate galaxy cluster diffuse emissions.

References
----------
.. [arnaud2005]
   Arnaud, Pointecouteau & Pratt 2005, A&A, 441, 893;
   http://adsabs.harvard.edu/abs/2005A%26A...441..893

.. [beck2005]
   Beck & Krause 2005, AN, 326, 414
   http://adsabs.harvard.edu/abs/2005AN....326..414B

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

.. [miniati2015]
   Miniati & Beresnyak 2015, Nature, 523, 59
   http://adsabs.harvard.edu/abs/2015Natur.523...59M

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
from scipy import optimize

from ...share import COSMO
from ...utils.units import (Units as AU,
                            Constants as AC,
                            UnitConversions as AUC)
from ...utils.draw import circle
from ...utils.transform import circle2ellipse


logger = logging.getLogger(__name__)


def beta_model(rho0, rc, beta):
    """
    Return a function that calculates the value (gas density) at a radius
    according to the β-model with the given parameters.
    """
    def func(r):
        x = r / rc
        return rho0 * (1 + x**2) ** (-1.5*beta)

    return func


def calc_gas_density_profile(mass, z, f_rc=0.1, beta=0.8):
    """
    Calculate the parameters of the β-model that is used to describe the
    gas density profile.

    NOTE
    ----
    The core radius is assumed to be: ``rc = 0.1 r_vir``, and the beta
    parameters is assumed to be ``β = 0.8``.

    Reference: [cassano2005],Sec.(4.1)

    Parameters
    ----------
    f_rc : float
        The fraction of the core radius to the virial radius.
        Default: 0.1
    beta : float
        The slope parameter of the β-model.
        Default: 0.8

    Returns
    -------
    fbeta : function
        A function of the β-model.
        Unit: [Msun/kpc^3]
    """
    r_vir = radius_virial(mass, z)  # [kpc]
    rc = f_rc * r_vir
    fint = beta_model(1, rc, beta)
    v = integrate.quad(lambda r: fint(r) * r**2,
                       a=0, b=r_vir)[0]  # [kpc^3]
    rho0 = mass * COSMO.baryon_fraction / (4*np.pi * v)  # [Msun/kpc^3]
    return beta_model(rho0, rc, beta)


def radius_overdensity(mass, overdensity, z=0.0):
    """
    Calculate the radius within which the mean density is ``overdensity``
    times of the cosmological critical density.

    Parameters
    ----------
    mass : float, `~numpy.ndarray`
        Total mass of the cluster
        Unit: [Msun]
    overdensity : float
        The times of density over the cosmological critical density,
        e.g., 200, 500.
    z : float, `~numpy.ndarray`, optional
        Redshift
        Default: 0.0 (i.e., present day)

    Returns
    -------
    radius : float, `~numpy.ndarray`
        Unit: [kpc]
    """
    rho = COSMO.rho_crit(z)  # [g/cm^3]
    r = (3*mass*AUC.Msun2g / (4*np.pi * overdensity * rho))**(1/3)  # [cm]
    return r * AUC.cm2kpc  # [kpc]


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
    return radius_overdensity(mass, overdensity=Dc, z=z)


def radius_cluster(mass, z=0):
    """
    Calculate the radius of the cluster.

    NOTE/XXX
    --------
    The cosmic evolution makes the whole thing too complicated, e.g.,
    the thermal energy density becomes more sensitive to the redshifts
    rather than cluster mass, therefore, the magnetic field, which is
    assumed to be correlated with the thermal energy density, increases
    more rapidly with increasing redshifts rather than cluster mass.
    This is not expected.  So ignore the cosmic evolution for modeling
    radio halos, keeping the things simple.

    Unit: [kpc]
    """
    return radius_virial(mass, z=0)


def radius_stripping(M_main, M_sub, z, f_rc=0.1, beta=0.8):
    """
    Calculate the stripping radius of the in-falling sub-cluster, which
    is determined by the equipartition between the static and ram pressure.

    Reference: [cassano2005],Eqs.(11,12,13,14)

    Returns
    -------
    rs : float
        The stripping radius of the sub-cluster.
        Unit: [kpc]
    """
    r_vir = radius_virial(M_sub, z)  # [kpc]
    rho_main = density_number_thermal(M_main, z) * AC.mu*AC.u  # [g/cm^3]
    f_rho_sub = calc_gas_density_profile(M_sub, z, f_rc, beta)  # [Msun/kpc^3]
    vi = velocity_impact(M_main, M_sub, z)  # [km/s]
    kT_sub = kT_cluster(M_sub, z)  # [keV]
    rhs = rho_main * vi**2 * AC.mu*AC.u / kT_sub  # [g/cm^3][g*km^2/s^2/keV]
    rhs *= 1e3 * AUC.J2erg / AUC.keV2erg  # [g/cm^3]
    rhs *= AUC.g2Msun / AUC.cm2kpc**3  # [Msun/kpc^3]
    try:
        rs = optimize.brentq(lambda r: f_rho_sub(r) - rhs,
                             a=0.1*r_vir, b=r_vir, xtol=1e-1)
    except ValueError:
        rs = 2*f_rc * r_vir
    return rs  # [kpc]


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


def kT_cluster(mass, z=0.0, radius=None, kT_out=0):
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


def density_gas(mass, z=0.0):
    """
    Calculate the mean gas density.
    Unit: [g/cm^3]
    """
    return density_number_thermal(mass, z) * AC.mu*AC.u  # [g/cm^3]


def density_energy_thermal(mass, z=0.0, kT_out=0):
    """
    Calculate the thermal energy density of the ICM.

    Returns
    -------
    e_th : float
        Energy density of the ICM
        Unit: [erg/cm^3]
    """
    n_th = density_number_thermal(mass=mass, z=z)  # [cm^-3]
    kT = kT_cluster(mass, z, kT_out=kT_out) * AUC.keV2erg  # [erg]
    e_th = (3.0/2) * kT * n_th
    return e_th


def density_energy_electron(n_e, gamma):
    """
    Calculate the energy density of relativistic electrons.

    Parameters
    ----------
    n_e : 1D float `~numpy.ndarray`
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
    e_spec = n_e * gamma*AU.mec2
    return integrate.simps(e_spec * gamma, np.log(gamma))  # in log grid


def density_number_electron(n_e, gamma):
    """
    Calculate the electron number density of the given spectrum.
    Unit: [cm^-3]
    """
    return integrate.simps(n_e * gamma, np.log(gamma))  # in log grid


def magnetic_field(mass, z, eta_b, kT_out=0):
    """
    Calculate the mean magnetic field strength within the ICM, which is
    also assumed to be uniform, according to the assumed fraction of the
    the magnetic field energy density w.r.t. the ICM thermal energy density.

    NOTE
    ----
    Magnetic field energy density: u_B = B^2 / (8π),
    where "B" in units of [G], then "u_B" has unit of [erg/cm^3].

    NOTE
    ----
    Magnetic fields and cosmic rays are strongly coupled and exchange
    energy.  Therefore equipartition between them is assumed, i.e.,
    X_cr (= ε_cr / ε_th) = η_b (= ε_b / ε_th)

    Reference: [beck2005],App.A

    Returns
    -------
    B : float
        The mean magnetic field strength within the ICM.
        Unit: [uG]
    """
    e_th = density_energy_thermal(mass=mass, z=z, kT_out=kT_out)
    B = np.sqrt(8*np.pi * eta_b * e_th) * 1e6  # [G] -> [uG]
    return B


def plasma_beta(mass, z, eta_b, kT_out=0):
    """
    Calculate the β value of the ICM, which is defined as:
        β ≡ P_gas / u_B
    where "P_gas" is the gas pressue: P_gas = n_th * kT;
    "u_B" is the magnetic field energy density: u_B = B² / 8π .

    Reference: Ref.[miniati2015],Eq.(2)
    """
    n_th = density_number_thermal(mass, z)  # [cm^-3]
    kT = kT_cluster(mass, z, kT_out=kT_out) * AUC.keV2erg  # [erg]
    P = n_th * kT
    B = magnetic_field(mass, z, eta_b=eta_b, kT_out=kT_out) * 1e-6  # [G]
    beta = 8*np.pi * P / B**2
    return beta


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
    Calculate the virial velocity, i.e., circular velocity at the
    virial radius.

    Unit: [km/s]
    """
    R_vir = radius_virial(mass, z) * AUC.kpc2cm  # [cm]
    vv = np.sqrt(AC.G * mass*AUC.Msun2g / R_vir)  # [cm/s]
    return vv / AUC.km2cm  # [km/s]


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
    return vi / AUC.km2cm  # [km/s]


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
