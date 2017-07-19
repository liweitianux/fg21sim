# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate the extended radio emissions from the galaxy cluster,
e.g., giant radio halos, radio relics.

NOTE
----
There are other types of extended radio emissions not considered
yet, e.g., mini-halos, roundish radio relics, etc.
"""

import logging

import numpy as np
import pandas as pd

from .formation import ClusterFormation
from .halo import RadioHalo
from ...sky import get_sky
from ...utils.cosmology import Cosmology
from ...utils.io import dataframe_to_csv


logger = logging.getLogger(__name__)


class GalaxyClusters:
    """
    Simulate the extended radio emissions from the galaxy clusters.

    NOTE
    ----
    Currently, only the *giant radio halos* are considered, while
    other types of extended emissions are missing, e.g., mini-halos,
    elongated relics, roundish relics.
    """
    # Component name
    name = "galaxy clusters"

    def __init__(self, configs):
        self.configs = configs
        self.sky = get_sky(configs)
        self._set_configs()

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = "extragalactic/clusters"
        self.catalog_path = self.configs.get_path(comp+"/catalog")
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        if self.sky.type_ == "patch":
            self.resolution = self.sky.pixelsize
        else:
            raise NotImplementedError("TODO: full-sky simulations")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        # Cosmology model
        self.H0 = self.configs.getn("cosmology/H0")
        self.OmegaM0 = self.configs.getn("cosmology/OmegaM0")
        self.Omegab0 = self.configs.getn("cosmology/Omegab0")
        self.sigma8 = self.configs.getn("cosmology/sigma8")
        self.cosmo = Cosmology(H0=self.H0, Om0=self.OmegaM0,
                               Ob0=self.Omegab0, sigma8=self.sigma8)
        #
        logger.info("Loaded and set up configurations")

    def _load_catalog(self):
        """
        Load the sampled (z, mass) catalogs from the Press-Schechter
        formalism for the clusters in this sky patch.

        Catalog columns
        ---------------
        * ``z`` : redshifts
        * ``mass`` : cluster mass; unit: [Msun]
        """
        self.catalog = pd.read_csv(self.catalog_path, comment="#")
        self.catalog_comment = [
            "z : redshift",
            "mass : cluster total mass [Msun]",
        ]
        num = len(self.catalog)
        logger.info("Loaded (z, mass) catalog: %d clusters" % num)

    def _process_catalog(self):
        """
        Do some basic processes to the catalog:

        * Generate random positions within the sky for each cluster;
        * Generate random elongated fraction;
        * Generate random rotation angle.

        Catalog columns
        ---------------
        * ``lon`` : longitudes; unit: [deg]
        * ``lat`` : latitudes; unit: [deg]
        * ``felong`` : elongated fraction, defined as the ratio of
                       elliptical semi-major axis to semi-minor axis;
                       restricted within [0.3, 1.0]
        * ``rotation`` : rotation angle; uniformly distributed within
                         [0, 360.0); unit: [deg]

        NOTE
        ----
        felong (elongated fraction) ::
            Adopt a definition (felong = b/a) similar to the Hubble
            classification for the elliptical galaxies.  As for the
            elliptical galaxies classification, E7 is the limit (e.g.,
            Wikipedia), therefore felong is also restricted within
            [0.3, 1.0], and sampled from a cut and absolute normal
            distribution centered at 1.0 with sigma ~0.7/3 (<= 3σ).
        """
        logger.info("Preliminary processes to the catalog ...")
        num = len(self.catalog)
        lon, lat = self.sky.random_points(n=num)
        self.catalog["lon"] = lon
        self.catalog["lat"] = lat
        self.catalog_comment.append(
            "lon, lat : longitudes and latitudes [deg]")
        logger.info("Added catalog columns: lon, lat.")

        felong_min = 0.3
        sigma = (1.0 - felong_min) / 3.0
        felong = 1.0 - np.abs(np.random.normal(scale=sigma, size=num))
        felong[felong < felong_min] = felong_min
        self.catalog["felong"] = felong
        self.catalog_comment.append(
            "felong : elongated fraction (= b/a)")
        logger.info("Added catalog column: felong.")

        rotation = np.random.uniform(low=0.0, high=360.0, size=num)
        self.catalog["rotation"] = rotation
        self.catalog_comment.append(
            "rotation : ellipse rotation angle [deg]")
        logger.info("Added catalog column: rotation.")

    def postprocess(self):
        """
        Do some necessary post-simulation operations.
        """
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the effective/inuse clusters catalog
        logger.info("Save the resulting catalog ...")
        if self.catalog_outfile is None:
            logger.warning("Catalog output file not set; skip saving!")
        else:
            dataframe_to_csv(self.catalog, outfile=self.catalog_outfile,
                             comment=self.catalog_comment,
                             clobber=self.clobber)
