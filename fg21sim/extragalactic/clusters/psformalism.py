# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Press-Schechter (PS) formalism

First determine the number of clusters within a sky patch (i.e., sky
coverage) according to the cluster distribution predicted by the PS
formalism; then sampling from the PS mass function to derive the mass
and redshift for each cluster.
"""

import logging
import random

import numpy as np
import pandas as pd

from ...share import CONFIGS, COSMO
from ...utils.interpolate import bilinear
from ...utils.units import UnitConversions as AUC


logger = logging.getLogger(__name__)


class PSFormalism:
    """
    Press-Schechter (PS) formalism

    Simulate the clusters number and their distribution (mass and z)
    within a sky patch of certain coverage.
    """
    def __init__(self, configs=CONFIGS):
        self.configs = configs
        self._set_configs()
        self._load_data()

    def _set_configs(self):
        """
        Load the required configurations and set them.
        """
        comp = "extragalactic/clusters"
        self.datafile = self.configs.get_path(comp+"/ps_data")
        self.f_darkmatter = self.configs.getn(comp+"/f_darkmatter")
        self.Mmin_cluster = self.configs.getn(comp+"/mass_min")  # [Msun]

    @property
    def Mmin_halo(self):
        return self.Mmin_cluster * self.f_darkmatter

    def _load_data(self, filepath=None):
        """
        Load dndM data and reformat into a 2D density grid together with
        redshifts and masses vectors.

        Data File Description
        ---------------------
        z1  mass1  density1
        z1  mass2  density2
        z1  ..     density3
        z2  mass1  density4
        z2  mass2  density5
        z2  ..     density6
        ...

        where,
        * Redshifts: 0.0 -> 3.02, even-spacing, step 0.02
        * Mass: unit 1e12 -> 9.12e15 [Msun], log-even (dark matter)
        * Density: [number]/dVc/dM
          with,
          - dVc: differential comvoing volume, [Mpc^3]/[sr]/[unit redshift]
        """
        if filepath is None:
            filepath = self.datafile
        data = np.loadtxt(filepath)
        redshifts = data[:, 0]
        masses = data[:, 1]
        densities = data[:, 2]

        redshifts = np.array(list(set(redshifts)))
        redshifts.sort()
        masses = np.array(list(set(masses)))
        masses.sort()
        densities = densities.reshape((len(redshifts), len(masses)))

        logger.info("Loaded PS data from file: %s" % filepath)
        logger.info("Number of redshift bins: %d" % len(redshifts))
        logger.info("Number of mass bins: %d" % len(masses))
        self.redshifts = redshifts
        self.masses = masses
        self.densities = densities

    @staticmethod
    def delta(x, logeven=False):
        """
        Calculate the delta values for each element of a vector,
        assuming they are evenly or log-evenly distributed,
        with extrapolating.
        """
        x = np.asarray(x)
        if logeven:
            x = np.log(x)
        step = x[1] - x[0]
        x1 = np.concatenate([[x[0]-step], x[:-1]])
        x2 = np.concatenate([x[1:], [x[-1]+step]])
        dx = (x2 - x1) * 0.5
        if logeven:
            dx = np.exp(dx)
        return dx

    @property
    def number_grid(self):
        """
        Calculate the number distribution w.r.t. redshift, mass, and
        unit coverage [sr] from the density distribution.
        """
        dz = self.delta(self.redshifts)
        dM = self.delta(self.masses)
        dMgrip, dzgrip = np.meshgrid(dM, dz)
        Mgrip, zgrip = np.meshgrid(self.masses, self.redshifts)
        dVcgrip = COSMO.dVc(zgrip)  # [Mpc^3/sr]
        numgrid = self.densities * dVcgrip * dzgrip * dMgrip
        return numgrid

    def calc_cluster_counts(self, coverage):
        """
        Calculate the total number of clusters (>= minimum mass) within
        the FoV coverage according to the number density distribution
        (e.g., predicted by the Press-Schechter mass function)

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
        logger.info("Determine the total number of clusters within "
                    "sky patch of coverage %.1f [deg^2]" % coverage)
        coverage *= AUC.deg2rad**2  # [deg^2] -> [rad^2] = [sr]
        midx = (self.masses >= self.Mmin_halo)
        numgrid = self.number_grid
        counts = np.sum(numgrid[:, midx]) * coverage
        self.counts = int(np.round(counts))
        logger.info("Total number of clusters: %d" % self.counts)
        return self.counts

    def sample_z_m(self, counts=None):
        """
        Randomly generate the requested number of pairs of (z, M) following
        the specified number distribution.

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
                    logger.debug("[%d/%d] %.1f%% done ..." %
                                 (i, counts, 100*i/counts))
        logger.info("Sampled %d pairs of (z, mass) for each cluster" % counts)

        df = pd.DataFrame(np.column_stack([z_list, M_list]),
                          columns=["z", "mass"])
        df["mass"] /= self.f_darkmatter
        comment = [
            "cluster number counts : %d" % counts,
            "z : redshift",
            "mass : cluster total mass [Msun]",
        ]
        self.clusters = df
        self.clusters_comment = comment
        return (df, comment)
