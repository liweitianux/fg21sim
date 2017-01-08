# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license

"""
Simulate cluster formation (i.e., merging history) using the extended
Press-Schechter formalism.

References
----------
[1] Randall, Sarazin & Ricker 2002, ApJ, 577, 579
    http://adsabs.harvard.edu/abs/2002ApJ...577..579R
[2] Cassano & Brunetti 2005, MNRAS, 357, 1313
    http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
"""

import logging

import numpy as np
import scipy.integrate
import scipy.special
import scipy.optimize

from .cosmology import Cosmology
from .mergertree import MergerTree


logger = logging.getLogger(__name__)


class ClusterFormation:
    """
    Simulate the cluster formation (i.e., merging history) using the extended
    Press-Schechter formalism by Monte Carlo methods.

    References
    ----------
    [1] Randall, Sarazin & Ricker 2002, ApJ, 577, 579
        http://adsabs.harvard.edu/abs/2002ApJ...577..579R
    [2] Cassano & Brunetti 2005, MNRAS, 357, 1313
        http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C

    Parameters
    ----------
    M0 : float
        Present-day (z=0) mass (unit: Msun) of the cluster.
    configs : `ConfigManager`
        A `ConfigManager` instance containing default and user configurations.
        For more details, see the example configuration specifications.

    Attributes
    ----------
    cosmo : `~Cosmology`
        Adopted cosmological model with custom utility functions.
    mtree : `~MergerTree`
        Merging history of this cluster.
    """
    def __init__(self, M0, configs):
        self.M0 = M0  # [Msun]
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """
        Set up the necessary class attributes according to the configs.
        """
        comp = "extragalactic/halos"
        # Minimum mass change (unit: Msun) of the main-cluster for a merger
        self.merger_mass_min = self.configs.getn(comp+"/merger_mass_min")
        # Cosmology model
        self.H0 = self.configs.getn("cosmology/H0")
        self.OmegaM0 = self.configs.getn("cosmology/OmegaM0")
        self.sigma8 = self.configs.getn("cosmology/sigma8")
        self.cosmo = Cosmology(H0=self.H0, Om0=self.OmegaM0,
                               sigma8=self.sigma8)
        logger.info("Loaded and set up configurations")

    @property
    def sigma_index(self):
        """
        The power-law spectral index assumed for the following density
        perturbations sigma(M).

        References: Ref.[1],Eq.(2)
        """
        n = -7/5
        alpha = (n+3) / 6
        return alpha

    def f_sigma(self, mass):
        """
        Current rms density fluctuations within a sphere of specified
        mass (unit: Msun).

        It is generally sufficient to consider a power-law spectrum of
        density perturbations, which is consistent with the CDM models.

        References: Ref.[1],Eq.(2)
        """
        alpha = self.sigma_index
        sigma = self.cosmo.sigma8 * (mass / self.cosmo.M8) ** (-alpha)
        return sigma

    def f_delta_c(self, z):
        """
        w = delta_c(z) is the critical linear overdensity for a region
        to collapse at redshift z.

        This is a monotone decreasing function.

        References: Ref.[1],App.A,Eq.(A1)
        """
        return self.cosmo.overdensity_crit(z)

    def f_dw_max(self, mass):
        """
        Calculate the allowed maximum step size for tracing cluster
        formation, therefore, the adopted step size is chosen to be half
        of this maximum value.

        dw^2 ~< abs(d(ln(sigma(M)^2)) / d(ln(M))) * (dMc / M) * sigma(M)^2
              = 2 * alpha * sigma(M)^2 * dMc / M

        References: Ref.[1],Sec.(3.1),Para.(1)
        """
        alpha = self.sigma_index
        dMc = self.merger_mass_min
        return np.sqrt(2 * alpha * self.f_sigma(mass)**2 * dMc / mass)

    def calc_z(self, delta_c):
        """
        Solve the redshift from the specified delta_c (a.k.a. w).
        """
        z = scipy.optimize.newton(
            lambda x: self.f_delta_c(x) - delta_c,
            x0=0, tol=1e-5)
        return z

    def calc_mass(self, S):
        """
        Calculate the mass corresponding to the given S.

        S = sigma(M)^2

        References: Ref.[1],Sec.(3)
        """
        alpha = self.sigma_index
        mass = self.cosmo.M8 * (S / self.cosmo.sigma8**2)**(-1/(2*alpha))
        return mass

    @staticmethod
    def cdf_K(dS, dw):
        """
        The cumulative probability distribution function of sub-cluster
        masses.

        References: Ref.[1],Eq.(5)
        """
        p = scipy.special.erfc(dw / np.sqrt(2*dS))
        return p

    @staticmethod
    def cdf_K_inv(p, dw):
        """
        Inverse function of the above ``cdf_K()``.
        """
        dS = 0.5 * (dw / scipy.special.erfcinv(p))**2
        return dS

    def gen_dS(self, dw, size=None):
        """
        Randomly generate values of dS by sampling the CDF ``cdf_K()``.
        """
        r = np.random.uniform(size=size)
        dS = self.cdf_K_inv(r, dw)
        return dS

    def simulate_mergertree(self):
        """
        Simulate the merger tree of this cluster by tracing its formation
        using the PS formalism.

        References: Ref.[1],Sec.(3.1)
        """
        self.mtree = self._trace_formation(self.M0, dMc=self.merger_mass_min)
        return self.mtree

    def _trace_formation(self, M, dMc, _z=None):
        """
        Recursively trace the cluster formation and thus simulate its
        merger tree.
        """
        z = 0.0 if _z is None else _z
        node_data = {"mass": M, "z": z, "age": self.cosmo.age(z)}

        if M <= dMc:
            # Stop the trace
            return MergerTree(data=node_data)

        # Trace the formation by simulate a merger/accretion event
        # Notation: progenitor (*1) -> current (*2)

        # Current properties
        w2 = self.f_delta_c(z=z)
        S2 = self.f_sigma(M) ** 2
        dw = 0.5 * self.f_dw_max(M)
        dS = self.gen_dS(dw)
        # Progenitor properties
        z1 = self.calc_z(w2 + dw)
        S1 = S2 + dS
        M1 = self.calc_mass(S1)
        dM = M - M1

        M_min = min(M1, dM)
        if M_min <= dMc:
            # Accretion
            M_new = M - M_min
            return MergerTree(
                data=node_data,
                main=self._trace_formation(M_new, dMc=dMc, _z=z1),
                sub=None
            )
        else:
            # Merger event
            M_main = max(M1, dM)
            M_sub = M_min
            return MergerTree(
                data=node_data,
                main=self._trace_formation(M_main, dMc=dMc, _z=z1),
                sub=self._trace_formation(M_sub, dMc=dMc, _z=z1)
            )
