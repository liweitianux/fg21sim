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

from .mergertree import MergerTree
from ...utils.cosmology import Cosmology


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
        Cluster mass at redshift z0
        Unit: [Msun]
    z0 : float
        Redshift from where to simulate former merging history.
    zmax : float, optional
        The maximum redshift/age when to stop the formation trace.
        Default: 3.0 (i.e., looking back time ~11.5 Gyr)
    ratio_major : float, optional
        The mass ratio of the main and sub clusters to define whether
        the merger is a major event or a  minor one.
        If ``M_main/M_sub < ratio_major``, then it is a major merger event.
        Default: 3.0
    cosmo : `~Cosmology`, optional
        Adopted cosmological model with custom utility functions.
    merger_mass_min : float, optional
        Minimum mass change to be regarded as a merger event instead of
        accretion.
        Unit: [Msun]

    Attributes
    ----------
    mtree : `~MergerTree`
        Merging history of this cluster.
    last_major_merger : dict, or None
        An dictionary containing the properties of the found last/recent
        major merger event, or None if not found.
    """
    def __init__(self, M0, z0, zmax=3.0, ratio_major=3.0,
                 cosmo=Cosmology(), merger_mass_min=1e12):
        self.M0 = M0  # [Msun]
        self.z0 = z0
        self.zmax = zmax
        self.ratio_major = ratio_major
        self.cosmo = cosmo
        self.merger_mass_min = merger_mass_min

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

    def simulate_mergertree(self, main_only=True):
        """
        Simulate the merger tree of this cluster by tracing its formation
        using the PS formalism.

        Parameters
        ----------
        main_only : bool, optional
            Whether to only trace the forming history of the main
            halo/cluster.
            (default: True)

        References: Ref.[1],Sec.(3.1)
        """
        logger.debug("Simulating cluster formation: " +
                     "M0={:.3g}[Msun] from z={:.3f} to z={zmax} ...".format(
                         self.M0, self.z0, zmax=self.zmax))
        self.main_only = main_only
        if main_only:
            logger.debug("Only trace the formation of the *main* cluster ...")
            self.mtree = self._trace_main()
        else:
            logger.debug("Trace formations of both main and sub cluster ...")
            self.mtree = self._trace_formation(self.M0, _z=self.z0)
        logger.debug("Simulated cluster formation with merger tree")
        return self.mtree

    @property
    def last_major_merger(self):
        """
        Identify and return the last major merger event

        Returns
        -------
        event :
            An dictionary containing the properties of the found major
            event:
            ``{"M_main": M_main, "M_sub": M_sub, "R_mass": R_mass,
               "z": z, "age": age}``;
            ``None`` if no major event found.
        """
        mtree = self.mtree
        event = None
        while mtree and mtree.main:
            if mtree.sub is None:
                mtree = mtree.main
                continue

            M_main = mtree.main.data["mass"]
            M_sub = mtree.sub.data["mass"]
            z = mtree.main.data["z"]
            age = mtree.main.data["age"]
            if M_main / M_sub < self.ratio_major:
                # Found a major merger event
                event = {"M_main": M_main,
                         "M_sub": M_sub,
                         "R_mass": M_main / M_sub,
                         "z": z,
                         "age": age}
                break

            # A minor merger event, continue
            mtree = mtree.main

        return event

    def _trace_main(self):
        """
        Iteratively trace the merger and accretion events of the
        main cluster/halo.
        """
        # Initial properties
        zc = self.z0
        Mc = self.M0
        mtree_root = MergerTree(data={"mass": Mc,
                                      "z": zc,
                                      "age": self.cosmo.age(zc)})
        logger.debug("[main] z=%.4f : mass=%g [Msun]" % (zc, Mc))

        mtree = mtree_root
        while True:
            # Whether to stop the trace
            if self.zmax is not None and zc > self.zmax:
                break
            if Mc <= self.merger_mass_min:
                break

            # Trace the formation by simulate a merger/accretion event
            # Notation: progenitor (*1) -> current (*2)

            # Current properties
            w2 = self.f_delta_c(z=zc)
            S2 = self.f_sigma(Mc) ** 2
            dw = 0.5 * self.f_dw_max(Mc)
            dS = self.gen_dS(dw)
            # Progenitor properties
            z1 = self.calc_z(w2 + dw)
            age1 = self.cosmo.age(z1)
            S1 = S2 + dS
            M1 = self.calc_mass(S1)
            dM = Mc - M1

            M_min = min(M1, dM)
            if M_min <= self.merger_mass_min:
                # Accretion
                M_main = Mc - M_min
                # NOTE: no sub node
            else:
                # Merger event
                M_main = max(M1, dM)
                M_sub = M_min
                mtree.sub = MergerTree(data={"mass": M_sub,
                                             "z": z1,
                                             "age": age1})
                logger.debug("[sub] z=%.4f : mass=%g [Msun]" % (z1, M_sub))

            # Update main cluster
            mtree.main = MergerTree(data={"mass": M_main,
                                          "z": z1,
                                          "age": age1})
            logger.debug("[main] z=%.4f : mass=%g [Msun]" % (z1, M_main))

            # Update for next iteration
            Mc = M_main
            zc = z1
            mtree = mtree.main

        return mtree_root

    def _trace_formation(self, M, _z=None, zmax=None):
        """
        Recursively trace the cluster formation and thus simulate its
        merger tree.
        """
        z = 0.0 if _z is None else _z
        node_data = {"mass": M, "z": z, "age": self.cosmo.age(z)}

        # Whether to stop the trace
        if self.zmax is not None and z > self.zmax:
            return MergerTree(data=node_data)
        if M <= self.merger_mass_min:
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
        if M_min <= self.merger_mass_min:
            # Accretion
            M_new = M - M_min
            return MergerTree(
                data=node_data,
                main=self._trace_formation(M_new, _z=z1),
                sub=None
            )
        else:
            # Merger event
            M_main = max(M1, dM)
            M_sub = M_min
            return MergerTree(
                data=node_data,
                main=self._trace_formation(M_main, _z=z1),
                sub=self._trace_formation(M_sub, _z=z1)
            )
