# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Press-Schechter (PS) formalism

First determine the number of clusters within a sky patch (i.e., sky
coverage) according to the cluster distribution predicted by the PS
formalism; then sampling from the halo mass function to derive the mass
and redshift for each cluster.
"""

import logging
import random

import numpy as np
import pandas as pd
import hmf

from ...share import CONFIGS, COSMO
from ...utils.interpolate import bilinear
from ...utils.units import UnitConversions as AUC
from ...utils.io import write_dndlnm


logger = logging.getLogger(__name__)


class PSFormalism:
    """
    Press-Schechter (PS) formalism

    Calculate the halo mass distribution with respect to mass and redshift,
    determine the clusters number counts and generate their distribution
    (mass and z) within a sky patch of certain coverage.
    """
    def __init__(self, configs=CONFIGS):
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """
        Load the required configurations and set them.
        """
        comp = "extragalactic/psformalism"
        self.model = self.configs.getn(comp+"/model")
        self.M_min = self.configs.getn(comp+"/M_min")
        self.M_max = self.configs.getn(comp+"/M_max")
        self.M_step = self.configs.getn(comp+"/M_step")
        self.z_min = self.configs.getn(comp+"/z_min")
        self.z_max = self.configs.getn(comp+"/z_max")
        self.z_step = self.configs.getn(comp+"/z_step")
        self.dndlnm_outfile = self.configs.get_path(comp+"/dndlnm_outfile")

        comp = "extragalactic/clusters"
        self.Mmin_cluster = self.configs.getn(comp+"/mass_min")  # [Msun]
        self.boost = self.configs.getn(comp+"/boost")

        self.clobber = self.configs.getn("output/clobber")

    @property
    def hmf_model(self):
        return {"PS": "PS",
                "SMT": "SMT",
                "JENKINS": "Jenkins"}[self.model.upper()]

    def hmf_massfunc(self, z=0.0):
        """
        Halo mass function as a `~hmf.MassFunction` instance.
        """
        if not hasattr(self, "_hmf_massfunc"):
            h = COSMO.h
            cosmo = COSMO._cosmo
            self._hmf_massfunc = hmf.MassFunction(
                Mmin=np.log10(self.M_min*h),
                Mmax=np.log10(self.M_max*h),
                dlog10m=self.M_step,
                hmf_model=self.hmf_model,
                cosmo_model=cosmo,
                n=COSMO.ns,
                sigma_8=COSMO.sigma8)
            logger.info("Initialized '%s' halo mass function." %
                        self.hmf_model)

        massfunc = self._hmf_massfunc
        massfunc.update(z=z)
        return massfunc

    @property
    def z(self):
        """
        The redshift points where to calculate the dndlnm data.
        """
        return np.arange(self.z_min, self.z_max+self.z_step/2, self.z_step)

    @property
    def mass(self):
        """
        The mass points where to calculate the dndlnm data.

        NOTE:
        The maximum mass end is exclusive, to be  consistent with hmf's
        mass function!
        """
        return 10 ** np.arange(np.log10(self.M_min),
                               np.log10(self.M_max),
                               self.M_step)

    @property
    def dndlnm(self):
        """
        The calculated halo mass distributions data.
        """
        if not hasattr(self, "_dndlnm"):
            self._dndlnm = self.calc_dndlnm()
        return self._dndlnm

    def calc_dndlnm(self):
        """
        Calculate the halo mass distributions expressed in ``dndlnm``,
        the differential mass distribution in terms of natural log of
        masses.
        Unit: [Mpc^-3] (the little "h" is folded into the values)

        NOTE
        ----
        dndlnm = d n(M,z) / d ln(M); [Mpc^-3]
        describes the number of halos per comoving volume (Mpc^3) at
        redshift z per unit logarithmic mass interval at mass M.
        """
        logger.info("Calculating dndlnm data ...")
        dndlnm = []
        h = COSMO.h
        for z_ in self.z:
            massfunc = self.hmf_massfunc(z_)
            dndlnm.append(massfunc.dndlnm * h**3)
        self._dndlnm = np.array(dndlnm)
        logger.info("Calculated dndlnm within redshift: %.1f - %.1f" %
                    (self.z_min, self.z_max))
        return self._dndlnm

    def write(self, outfile=None):
        """
        Write the calculate dndlnm data into file as NumPy ".npz" format.
        """
        if outfile is None:
            outfile = self.dndlnm_outfile
        write_dndlnm(outfile, dndlnm=self.dndlnm, z=self.z, mass=self.mass,
                     clobber=self.clobber)
        logger.info("Wrote dndlnm data into file: %s" % outfile)

    @property
    def Mmin_halo(self):
        return self.Mmin_cluster * COSMO.darkmatter_fraction

    @staticmethod
    def delta(x, logeven=False):
        """
        Calculate the delta values for each element of a vector,
        assuming they are evenly or log-evenly distributed,
        by extrapolating.
        """
        x = np.asarray(x)
        if logeven:
            ratio = x[1] / x[0]
            x_left = x[0] / ratio
            x_right = x[-1] * ratio
        else:
            step = x[1] - x[0]
            x_left = x[0] - step
            x_right = x[-1] + step
        x2 = np.concatenate([[x_left], x, [x_right]])
        dx = (x2[2:] - x2[:-2]) / 2
        return dx

    @property
    def number_grid(self):
        """
        The halo number per unit solid angle [sr] distribution w.r.t.
        mass and redshift.
        Unit: [/sr]
        """
        if not hasattr(self, "_number_grid"):
            dz = self.delta(self.z)
            dM = self.delta(self.mass, logeven=True)
            dlnM = dM / self.mass
            dlnMgrid, dzgrid = np.meshgrid(dlnM, dz)
            __, zgrid = np.meshgrid(self.mass, self.z)
            dVcgrid = COSMO.dVc(zgrid)  # [Mpc^3/sr]
            self._number_grid = self.dndlnm * dlnMgrid * (dVcgrid*dzgrid)

        return self._number_grid

    def calc_cluster_counts(self, coverage):
        """
        Calculate the total number of clusters (>= minimum mass) within
        the FoV coverage according to the halo number density distribution.

        Parameters
        ----------
        coverage : float
            The coverage of the sky patch within which to determine the
            total number of clusters.
            Unit: [deg^2]

        Returns
        -------
        counts : int
            The total number of clusters within the sky patch.

        Attributes
        ----------
        counts
        """
        logger.info("Calculating the total number of clusters within "
                    "sky patch of coverage %.1f [deg^2]" % coverage)
        coverage *= AUC.deg2rad**2  # [deg^2] -> [rad^2] = [sr]
        midx = (self.mass >= self.Mmin_halo)
        numgrid = self.number_grid
        counts = np.sum(numgrid[:, midx]) * coverage
        counts *= self.boost  # number boost factor
        self.counts = int(np.round(counts))
        logger.info("Total number of clusters: %d" % self.counts)
        return self.counts

    def sample_z_m(self, counts=None):
        """
        Randomly generate the requested number of pairs of (z, M) following
        the halo number distribution.

        Parameters
        ----------
        counts : int, optional
            The number of (z, mass) pairs to be sampled.
            If not specified, then default to ``self.counts``

        Returns
        -------
        df : `~pandas.DataFrame`
            A Pandas data frame with 2 columns, i.e., ``z`` and ``mass``.
        comment : list[str]
            Comments to the above data frame.

        Attributes
        ----------
        clusters : df
        clusters_comment : comment
        """
        if counts is None:
            counts = self.counts
        logger.info("Sampling (z, mass) pairs for %d clusters ..." % counts)

        redshifts = self.redshifts
        masses = self.masses
        zmin = redshifts.min()
        zmax = redshifts.max()
        Mmax = masses.max()
        midx = (masses >= self.Mmin_halo)
        numgrid = self.number_grid
        numgrid2 = numgrid[:, midx]
        NM = numgrid2.max()
        z_list = []
        M_list = []
        i = 0
        while i < counts:
            z = random.uniform(zmin, zmax)
            M = random.uniform(self.Mmin_halo, Mmax)
            r = random.random()
            zi1 = (self.redshifts < z).sum()
            zi2 = zi1 - 1
            if zi2 < 0:
                zi2 += 1
                zi1 += 1
            Mi1 = (self.masses < M).sum()
            Mi2 = Mi1 - 1
            if Mi2 < 0:
                Mi2 += 1
                Mi1 += 1
            N = bilinear(
                z, np.log(M),
                p11=(redshifts[zi1], np.log(masses[Mi1]), numgrid[zi1, Mi1]),
                p12=(redshifts[zi1], np.log(masses[Mi2]), numgrid[zi1, Mi2]),
                p21=(redshifts[zi2], np.log(masses[Mi1]), numgrid[zi2, Mi1]),
                p22=(redshifts[zi2], np.log(masses[Mi2]), numgrid[zi2, Mi2]))
            if r < N/NM:
                z_list.append(z)
                M_list.append(M)
                i += 1
                if i % 100 == 0:
                    logger.info("[%d/%d] %.1f%% ..." %
                                (i, counts, 100*i/counts))
        logger.info("Sampled %d pairs of (z, mass) for each cluster" % counts)

        df = pd.DataFrame(np.column_stack([z_list, M_list]),
                          columns=["z", "mass"])
        df["mass"] /= COSMO.darkmatter_fraction
        comment = [
            "dndM data: %s" % self.datafile,
            "cluster minimum mass: %.2e [Msun]" % self.Mmin_cluster,
            "dark matter fraction: %.2f" % COSMO.darkmatter_fraction,
            "cluster counts: %d" % counts,
            "boost factor for cluster counts: %s" % self.boost,
            "z - redshift",
            "mass - cluster total mass [Msun]",
        ]
        self.clusters = df
        self.clusters_comment = comment
        return (df, comment)
