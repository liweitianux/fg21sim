# Copyright (c) 2016-2017 Weitian LI <liweitianux@live.com>
# MIT license

"""
Flat ΛCDM cosmological model.
"""

import numpy as np
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM

from ..configs import configs
from .units import (UnitConversions as AUC, Constants as AC)


class Cosmology:
    """
    Flat ΛCDM cosmological model.

    Attributes
    ----------
    H0 : float
        Hubble parameter at present day (z=0)
    Om0 : float
        Density parameter of (dark and baryon) matter at present day
    Ob0 : float
        Density parameter of baryon at present day
    Ode0 : float
        Density parameter of dark energy at present day
    sigma8 : float
        Present-day rms density fluctuation on a scale of 8 h^-1 Mpc.

    References
    ----------
    [1] https://astro.uni-bonn.de/~pavel/WIKIPEDIA/Lambda-CDM_model.html
    [2] https://en.wikipedia.org/wiki/Lambda-CDM_model
    [3] Randall, Sarazin & Ricker 2002, ApJ, 577, 579
        http://adsabs.harvard.edu/abs/2002ApJ...577..579R
        Sec.(2)
    """
    def __init__(self, H0=configs.getn("cosmology/H0"),
                 Om0=configs.getn("cosmology/OmegaM0"),
                 Ob0=configs.getn("cosmology/Omegab0"),
                 sigma8=configs.getn("cosmology/sigma8")):
        self.H0 = H0  # [km/s/Mpc]
        self.Om0 = Om0
        self.Ob0 = Ob0
        self.Ode0 = 1.0 - Om0
        self.sigma8 = sigma8
        self._cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0)

    @property
    def h(self):
        """
        Dimensionless/reduced Hubble parameter
        """
        return self.H0 / 100.0

    @property
    def M8(self):
        """
        Mass contained in a sphere of radius of 8 h^-1 Mpc.
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
        """
        return self.H0 * self.E(z)

    def DL(self, z):
        """
        Luminosity distance at redshift z.
        Unit: [Mpc]
        """
        return self._cosmo.luminosity_distance(z).value

    def DA(self, z):
        """
        Angular diameter distance at redshift z.
        Unit: [Mpc]
        """
        return self._cosmo.angular_diameter_distance(z).value

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
        z : float
            Redshift

        Returns
        -------
        age : float
            Age of the universe (cosmic time) at the given redshift.
            Unit: [Gyr]

        References
        ----------
        [1] Thomas & Kantowski 2000, Physical Review D, 62, 103507
            http://adsabs.harvard.edu/abs/2000PhRvD..62j3507T
            Eq.(18)
        """
        t_H = self.hubble_time
        t = (t_H * (2/3/np.sqrt(1-self.Om0)) *
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
        age : float
            Age of the universe (cosmic time), unit [Gyr]

        Returns
        -------
        z : float
            Redshift corresponding to the specified age.
        """
        t_H = self.hubble_time
        term1 = (1/self.Om0) - 1
        term2 = np.sinh(3*age*np.sqrt(1-self.Om0) / (2*t_H)) ** 2
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

    def overdensity_virial(self, z):
        """
        Calculate the virial overdensity, which generally used to
        determine the virial radius of a cluster.

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eqs.(10,A4)
        """
        omega_z = (1 / self.Om(z)) - 1
        Delta_c = 18*np.pi**2 * (1 + 0.4093 * omega_z**0.9052)
        return Delta_c

    def overdensity_crit(self, z):
        """
        Critical (linear) overdensity for a region to collapse
        at a redshift z.

        References
        ----------
        [1] Randall, Sarazin & Ricker 2002, ApJ, 577, 579
            http://adsabs.harvard.edu/abs/2002ApJ...577..579R
            Appendix.A, Eq.(A1)
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

        References
        ----------
        [1] Randall, Sarazin & Ricker 2002, ApJ, 577, 579
            http://adsabs.harvard.edu/abs/2002ApJ...577..579R
            Appendix.A, Eq.(A7)
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
        if not hasattr(self, "_growth_factor0"):
            self._growth_factor0 = self.growth_factor(0)
        return self._growth_factor0
