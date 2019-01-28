# Copyright (c) 2017-2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Simulate cluster formation (i.e., merging history) using the extended
Press-Schechter formalism.

References
----------
.. [randall2002]
   Randall, Sarazin & Ricker 2002, ApJ, 577, 579
   http://adsabs.harvard.edu/abs/2002ApJ...577..579R

.. [cassano2005]
   Cassano & Brunetti 2005, MNRAS, 357, 1313
   http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
"""

import logging
import operator as op

import numpy as np
import scipy.integrate
import scipy.special
import scipy.optimize

from .mergertree import MergerTree
from ...share import COSMO


logger = logging.getLogger(__name__)


class ClusterFormation:
    """
    Simulate the cluster formation (i.e., merging history) by employing
    the extended Press-Schechter formalism.

    Parameters
    ----------
    M0 : float
        Cluster total mass at redshift ``z0``
        Unit: [Msun]
    z0 : float
        Redshift from where to simulate former merging history.
    zmax : float, optional
        The maximum redshift/age when to stop the formation trace.
        Default: 4.0 (i.e., looking back time ~12.1 Gyr)
    merger_mass_min : float, optional
        Minimum mass change to be regarded as a merger event instead of
        accretion.
        Unit: [Msun]
    """
    def __init__(self, M0, z0, zmax=4.0, merger_mass_min=1e12):
        self.M0 = M0  # [Msun]
        self.z0 = z0
        self.zmax = zmax
        self.merger_mass_min = merger_mass_min

    @property
    def sigma_index(self):
        """
        The power-law spectral index assumed for the following density
        perturbations sigma(M).

        References: Ref.[randall2002],Eq.(2)
        """
        n = -7/5
        alpha = (n+3) / 6
        return alpha

    def f_sigma(self, mass):
        """
        Current r.m.s. density fluctuations within a sphere of the given
        mean mass (unit: [Msun]).

        It is generally sufficient to consider a power-law spectrum of
        density perturbations, which is consistent with the CDM models.

        References: Ref.[randall2002],Eq.(2)
        """
        alpha = self.sigma_index
        sigma = COSMO.sigma8 * (mass / COSMO.M8) ** (-alpha)
        return sigma

    def f_delta_c(self, z):
        """
        w = delta_c(z) is the critical linear overdensity for a region
        to collapse at redshift z.

        This is a monotone decreasing function.

        References: Ref.[randall2002],App.A,Eq.(A1)
        """
        return COSMO.overdensity_crit(z)

    def f_dw_max(self, mass):
        """
        Calculate the allowed maximum step size for tracing cluster
        formation, therefore, the adopted step size is chosen to be half
        of this maximum value.

        dw^2 ~< abs(d(ln(sigma(M)^2)) / d(ln(M))) * (dMc / M) * sigma(M)^2
              = 2 * alpha * sigma(M)^2 * dMc / M

        References: Ref.[randall2002],Sec.(3.1),Para.(1)
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
        Calculate the mass corresponding to the given S, which is
        defined as: S = sigma(M)^2

        References: Ref.[randall2002],Sec.(3)
        """
        alpha = self.sigma_index
        mass = COSMO.M8 * (S / COSMO.sigma8**2)**(-1/(2*alpha))
        return mass

    @staticmethod
    def cdf_K(dS, dw):
        """
        The cumulative probability distribution function of sub-cluster
        masses.

        References: Ref.[randall2002],Eq.(5)
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

    def simulate_mtree(self, main_only=True):
        """
        Simulate the merger tree of this cluster by tracing its formation
        using the PS formalism.

        Parameters
        ----------
        main_only : bool, optional
            Whether to only trace the forming history of the main cluster.
            Default: True

        References: Ref.[randall2002],Sec.(3.1)
        """
        logger.debug("Simulating cluster formation: " +
                     "M0=%.3e[Msun] " % self.M0 +
                     "from z={z0:.3f} back to z={zmax} ...".format(
                         z0=self.z0, zmax=self.zmax))
        self.main_only = main_only
        if main_only:
            logger.debug("Only trace the formation of the *main* cluster ...")
            self.mtree = self._trace_main()
        else:
            logger.debug("Trace formations of both main and sub cluster ...")
            self.mtree = self._trace_formation(self.M0, _z=self.z0)
        logger.debug("Simulated cluster formation with merger tree.")
        return self.mtree

    def recent_major_merger(self, mtree=None, ratio_major=3.0):
        """
        Identify and return the most recent major merger event.

        Parameters
        ----------
        mtree : `~MergerTree`, optional
            Specify the merger tree from which to identify the most
            recent merger event.
            Default: self.mtree
        ratio_major : float, optional
            The mass ratio of the main and sub clusters to define whether
            the merger is a major event or a  minor one.
            If ``M_main/M_sub < ratio_major``, then it is a major merger.
            Default: 3.0

        Returns
        -------
        event : dict
            A dictionary with the properties of the found major event:
            ``{"M_main": M_main, "M_sub": M_sub, "R_mass": R_mass,
               "z": z, "age": age}``;
            ``{}`` if no major event found.
        """
        if mtree is None:
            mtree = self.mtree

        for main, sub in mtree.itermain():
            if main["mass"] <= sub.get("mass", 0) * ratio_major:
                event = {"M_main": main["mass"],
                         "M_sub": sub["mass"],
                         "R_mass": main["mass"] / sub["mass"],
                         "z": main["z"],
                         "age": main["age"]}
                return event
        return {}

    def maximum_merger(self, mtree=None):
        """
        The merger event corresponding to the biggest sub cluster, i.e.,
        the main cluster gains the most mass.

        NOTE
        ----
        Sometimes, the maximum merger event found here is not an major
        merger event.

        Returns
        -------
        event : dict
            Same as the above ``self.recent_major_event``.
            ``{}`` if no mergers occurred during the traced period.
        """
        if mtree is None:
            mtree = self.mtree

        event_max = {"M_main": 0, "M_sub": 0, "R_mass": 0, "z": -1, "age": -1}
        for main, sub in mtree.itermain():
            if sub.get("mass", -1) > event_max["M_sub"]:
                event_max = {"M_main": main["mass"],
                             "M_sub": sub["mass"],
                             "R_mass": main["mass"] / sub["mass"],
                             "z": main["z"],
                             "age": main["age"]}

        if event_max["z"] <= 0:
            logger.warning("No mergers occurred.")
            return {}
        else:
            return event_max

    def history(self, mtree=None, merger_only=False):
        """
        Extract and return all the formation events, e.g., merger and
        accretion.

        Parameters
        ----------
        mtree : `~MergerTree`, optional
            The simulated merger tree from which to extract the history.
            Default: ``self.mtree``
        merger_only : bool, optional
            If ``True``, only extract the merger events.

        Returns
        -------
        evlist : list[event]
            List of events with each element being a dictionary of the
            event properties.
        """
        if mtree is None:
            mtree = self.mtree

        evlist = []
        for main, sub in mtree.itermain():
            z, age, M_main = op.itemgetter("z", "age", "mass")(main)
            if sub:
                # merger
                M_sub = sub["mass"]
                R_mass = M_main / M_sub
            else:
                # accretion
                if merger_only:
                    continue
                M_sub, R_mass = None, None

            evlist.append({
                "z": z,
                "age": age,
                "M_main": M_main,
                "M_sub": M_sub,
                "R_mass": R_mass,
            })

        return evlist

    def mergers(self, mtree=None):
        """
        Extract and return all the merger events.
        """
        return self.history(mtree=mtree, merger_only=True)

    def _trace_main(self):
        """
        Iteratively trace the merger and accretion events of the main
        cluster/halo only.
        """
        # Initial properties
        zc = self.z0
        Mc = self.M0
        mtree_root = MergerTree(data={"mass": Mc,
                                      "z": zc,
                                      "age": COSMO.age(zc)})
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
            age1 = COSMO.age(z1)
            S1 = S2 + dS
            M1 = self.calc_mass(S1)
            dM = Mc - M1

            M_min = min(M1, dM)
            if M_min <= self.merger_mass_min:
                # Accretion
                M_main = Mc - M_min
                # NOTE: no sub node
            else:
                # Merger
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
        Recursively trace the cluster/halo formation and thus simulate
        its merger tree.
        """
        z = 0.0 if _z is None else _z
        node_data = {"mass": M, "z": z, "age": COSMO.age(z)}

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
            # Merger
            M_main = max(M1, dM)
            M_sub = M_min
            return MergerTree(
                data=node_data,
                main=self._trace_formation(M_main, _z=z1),
                sub=self._trace_formation(M_sub, _z=z1)
            )
