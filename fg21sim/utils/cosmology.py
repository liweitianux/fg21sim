# Copyright (c) 2016-2017,2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Cosmology calculator in a flat ΛCDM universe.

References
----------
.. [unibonn-wiki]
   https://astro.uni-bonn.de/~pavel/WIKIPEDIA/Lambda-CDM_model.html

.. [wikipedia-lcdm]
   https://en.wikipedia.org/wiki/Lambda-CDM_model

.. [randall2002]
   Randall, Sarazin & Ricker 2002, ApJ, 577, 579
   http://adsabs.harvard.edu/abs/2002ApJ...577..579R
   Sec.(2)

.. [hogg1999]
   Hogg 1999, arXiv:astro-ph/9905116
   http://adsabs.harvard.edu/abs/1999astro.ph..5116H

.. [thomas2000]
   Thomas & Kantowski 2000, Physical Review D, 62, 103507
   http://adsabs.harvard.edu/abs/2000PhRvD..62j3507T

.. [ellis2007]
   Ellis 2007, General Relativity and Gravitation, 39, 1047
   http://adsabs.harvard.edu/abs/2007GReGr..39.1047E

.. [cassano2005]
   Cassano & Brunetti 2005, MNRAS, 357, 1313
   http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C

.. [wikipedia-virial]
   https://en.wikipedia.org/wiki/Virial_mass

.. [bryan1998]
   http://adsabs.harvard.edu/abs/1998ApJ...495...80B
"""

import logging

import numpy as np
from scipy import integrate
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM

from .units import (UnitConversions as AUC, Constants as AC)


logger = logging.getLogger(__name__)


class Cosmology:
    """
    Flat ΛCDM cosmological model.

    Attributes
    ----------
    H0 : float
        Hubble parameter at present day (z=0)
        Unit: [km/s/Mpc]
    Om0 : float
        Density parameter of (dark and baryon) matter at present day
    Ob0 : float
        Density parameter of baryon at present day
    Ode0 : float
        Density parameter of dark energy at present day
    Tcmb0 : float
        Present-day CMB temperature
        Unit: [K]
    sigma8 : float
        Present-day rms density fluctuation on a scale of 8 h^-1 [Mpc]
    ns : float
        Scalar spectral index

    Internal attributes
    -------------------
    _cosmo : `~astropy.cosmology.Cosmology`
        Astropy cosmology instance to help calculations.
    _growth_factor0 : float
        Present day (z=0) growth factor
    """
    # Present day (z=0) growth factor
    _growth_factor0 = None

    def __init__(self, H0=71.0, Om0=0.27, Ob0=0.046,
                 Tcmb0=2.725, sigma8=0.81, ns=0.96):
        self.setup(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0, sigma8=sigma8, ns=ns)

    def setup(self, **kwargs):
        """
        Setup/update the parameters of the cosmology model.
        """
        for key, value in kwargs.items():
            if key in ["H0", "Om0", "Ob0", "Tcmb0", "sigma8", "ns"]:
                setattr(self, key, value)
            else:
                raise ValueError("unknown parameter: %s" % key)

        self.Ode0 = 1.0 - self.Om0
        self._cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, Ob0=self.Ob0,
                                    Tcmb0=self.Tcmb0)
        self._growth_factor0 = None
        logger.info("Setup cosmology with: {0}".format(kwargs))

    @property
    def h(self):
        """
        Dimensionless/reduced Hubble parameter
        """
        return self.H0 / 100.0

    @property
    def M8(self):
        """
        Mass contained in a sphere of radius of 8 h^-1 [Mpc].
        Unit: [Msun]
        """
        r = 8 * AUC.Mpc2cm / self.h  # [cm]
        M8 = (4*np.pi/3) * r**3 * self.rho_crit(0)  # [g]
        M8 *= AUC.g2Msun  # [Msun]
        return M8

    def E(self, z):
        """
        Redshift evolution factor.
        """
        return np.sqrt(self.Om0 * (1+z)**3 + self.Ode0)

    def H(self, z):
        """
        Hubble parameter at redshift z.
        Unit: [km/s/Mpc]
        """
        return self.H0 * self.E(z)

    def Dc(self, z):
        """
        Comoving distance at redshift z.
        Unit: [Mpc]
        """
        return self._cosmo.comoving_distance(z).value

    def Dc_to_redshift(self, Dc, zmin=0, zmax=3, zstep=0.01):
        """
        Calculate the redshifts corresponding to the given comoving
        distances by interpolation.

        Parameters
        ----------
        Dc : float, or `~numpy.ndarray`
            Comoving distances
            Unit: [Mpc]
        zmin, zmax : float, optional
            The minimum and maximum redshift within which the input
            comoving distances are enclosed; otherwise, a error will be
            raised during the calculation.
        zstep : float, optional
            The redshift step size adopted to do the interpolation.

        Returns
        -------
        redshift : float, or `~numpy.ndarray`
            Calculated redshifts w.r.t. the input comoving distances.

        Raises
        ------
        ValueError :
            The ``zmin`` or ``zmax`` is not enough to enclose the input
            comoving distance range.
        """
        Dc_min, Dc_max = self.Dc([zmin, zmax])  # [Mpc]
        if np.sum(Dc < Dc_min) > 0:
            raise ValueError("zmin=%s is too big for input Dc" % zmin)
        if np.sum(Dc > Dc_max) > 0:
            raise ValueError("zmax=%s is too small for input Dc" % zmax)

        z_ = np.arange(zmin, zmax+zstep/2, zstep)
        Dc_ = self.Dc(z_)
        Dc_interp = interpolate.interp1d(Dc_, z_, kind="linear")
        return Dc_interp(Dc)

    def DA(self, z):
        """
        Angular diameter distance at redshift z.
        Unit: [Mpc]

        Defined as the ratio of an object's physical transverse size
        to its (observed) angular size (in radians).  It is used to
        convert the observed angular separations between sources into
        their proper separations.

        NOTE
        ----
        This distance is NOT increasing indefinitely as z -> ∞.

        Reference: Ref.[hogg1999]
        """
        return self._cosmo.angular_diameter_distance(z).value

    def DL(self, z):
        """
        Luminosity distance at redshift z.
        Unit: [Mpc]

        Defined by the relationship between the measured bolometric
        (i.e., integrated over all frequencies) flux S_bolo and the
        object's intrinsic bolometric luminosity L_bolo.

        NOTE
        ----
        DL = DA * (1+z)^2
        This is the general reciprocity theorem in General Relativity.

        Reference
        ---------
        * Ref.[hogg1999],Eq.(20,21)
        * Ref.[ellis2007]
        """
        return self._cosmo.luminosity_distance(z).value

    @property
    def hubble_time(self):
        """
        Hubble time.
        Unit: [Gyr]
        """
        uconv = AUC.Mpc2km * AUC.s2Gyr
        t_H = (1.0/self.H0) * uconv  # [Gyr]
        return t_H

    def age(self, z):
        """
        Cosmic time (age) at redshift z.

        Parameters
        ----------
        z : `~numpy.ndarray`
            Redshift

        Returns
        -------
        age : `~numpy.ndarray`
            Age of the universe (cosmic time) at the given redshift.
            Unit: [Gyr]

        References: Ref.[thomas2000],Eq.(18)
        """
        z = np.asarray(z)
        t_H = self.hubble_time
        t = ((2*t_H / 3 / np.sqrt(1-self.Om0)) *
             np.arcsinh(np.sqrt((1/self.Om0 - 1) / (1+z)**3)))
        return t

    @property
    def age0(self):
        """
        Present age of the universe.
        """
        return self.age(0)

    def redshift(self, age):
        """
        Invert the above ``self.age(z)`` formula analytically, to calculate
        the redshift corresponding to the given cosmic time (age).

        Parameters
        ----------
        age : `~numpy.ndarray`
            Age of the universe (i.e., cosmic time)
            Unit: [Gyr]

        Returns
        -------
        z : `~numpy.ndarray`
            Redshift corresponding to the specified age.
        """
        age = np.asarray(age)
        t_H = self.hubble_time
        term1 = (1/self.Om0) - 1
        term2 = np.sinh(3*age * np.sqrt(1-self.Om0) / (2*t_H)) ** 2
        z = (term1 / term2) ** (1/3) - 1
        return z

    def rho_crit(self, z):
        """
        Critical density at redshift z.
        Unit: [g/cm^3]
        """
        rho = 3 * self.H(z)**2 / (8*np.pi * AC.G)
        rho *= AUC.km2Mpc**2
        return rho

    def Om(self, z):
        """
        Density parameter of matter at redshift z.
        """
        return self.Om0 * (1+z)**3 / self.E(z)**2

    @property
    def baryon_fraction(self):
        """
        The cosmological mean baryon fraction (w.r.t. matter).

        XXX: assumed to be *constant* regardless of redshifts!
        """
        return self.Ob0 / self.Om0

    @property
    def darkmatter_fraction(self):
        """
        The cosmological mean dark matter fraction (w.r.t. matter).
        """
        return 1 - self.baryon_fraction

    def overdensity_virial(self, z):
        """
        Calculate the virial overdensity, which generally used to
        determine the virial radius of a cluster.

        References
        ----------
        * Ref.[bryan1998],Eqs.(5,6)
        * Ref.[wikipedia-virial]
        """
        x = self.Om(z) - 1
        return 18*np.pi**2 + 82*x - 39 * x**2

    def overdensity_crit(self, z):
        """
        Critical (linear) overdensity for a region to collapse
        at a redshift z.

        References: Ref.[randall2002],Eq.(A1)
        """
        coef = 3 * (12*np.pi) ** (2/3) / 20
        D0 = self.growth_factor0
        D_z = self.growth_factor(z)
        Om_z = self.Om(z)
        delta_c = coef * (D0 / D_z) * (1 + 0.0123*np.log10(Om_z))
        return delta_c

    def growth_factor(self, z):
        """
        Growth factor at redshift z.

        References: Ref.[randall2002],Eq.(A7)
        """
        x0 = (2 * self.Ode0 / self.Om0) ** (1/3)
        x = x0 / (1 + z)
        coef = np.sqrt(x**3 + 2) / (x**1.5)
        integral = integrate.quad(lambda y: y**1.5 * (y**3+2)**(-1.5),
                                  a=0, b=x, epsabs=1e-5, epsrel=1e-5)[0]
        D = coef * integral
        return D

    @property
    def growth_factor0(self):
        """
        Present-day (z=0) growth factor.
        """
        if self._growth_factor0 is None:
            self._growth_factor0 = self.growth_factor(0)
        return self._growth_factor0

    def dVc(self, z):
        """
        Calculate the differential comoving volume.

        The dimensions is [Mpc^3]/[sr]/[unit redshift].
        """
        return self._cosmo.differential_comoving_volume(z).value
