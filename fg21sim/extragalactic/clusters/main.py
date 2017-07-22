# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate the extended radio emissions from the galaxy cluster,
e.g., giant radio halos, radio relics.

NOTE
----
There are other types of extended radio emissions not considered
yet, e.g., mini-halos, roundish radio relics, etc.

References
----------
.. [cassano2012]
   Cassano et al. 2012, A&A, 548, A100
   http://adsabs.harvard.edu/abs/2012A%26A...548A.100C
"""

import logging

import numpy as np
import pandas as pd

from .formation import ClusterFormation
from .halo import RadioHalo
from ...sky import get_sky
from ...utils import cosmo
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

    Attributes
    ----------
    configs : `~ConfigManager`
        A `ConfigManager` instance containing default and user configurations.
        For more details, see the example configuration specifications.
    halo_configs : dict
        A dictionary containing the configurations for halo simulation.
    sky : `~SkyPatch` or `SkyHealpix`
        The sky instance to deal with the simulation sky as well as the
        output map.
        XXX: current full-sky HEALPix map is NOT supported!
    """
    # Component name
    name = "galaxy clusters (halos)"

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
        self.merger_mass_min = self.configs.getn(comp+"/merger_mass_min")
        self.ratio_major = self.configs.getn(comp+"/ratio_major")
        self.tau_merger = self.configs.getn(comp+"/tau_merger")

        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")

        # Sky and resolution
        if self.sky.type_ == "patch":
            self.resolution = self.sky.pixelsize  # [arcsec]
        else:
            raise NotImplementedError("TODO: full-sky simulations")

        logger.info("Loaded and set up configurations")

    @property
    def halo_configs(self):
        """
        Configurations for radio halo simulation as a dictionary.
        """
        comp = "extragalactic/halos"
        haloconf = {
            "eta_turb": self.configs.getn(comp+"/eta_turb"),
            "eta_e": self.configs.getn(comp+"/eta_e"),
            "gamma_min": self.configs.getn(comp+"/gamma_min"),
            "gamma_max": self.configs.getn(comp+"/gamma_max"),
            "gamma_np": self.configs.getn(comp+"/gamma_num"),
            "buffer_np": self.configs.getn(comp+"/buffer_np"),
            "time_step": self.configs.getn(comp+"/time_step"),
            "injection_index": self.configs.getn(comp+"/injection_index"),
        }
        return haloconf

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

    def _simulate_mergers(self):
        """
        Simulate the *recent major merger* event for each cluster.

        First simulate the cluster formation history by tracing the
        merger and accretion events of the main cluster, then identify
        the most recent major merger event according to the mass ratio
        of two merging clusters.  And the properties of the found merger
        event are appended to the catalog.

        NOTE
        ----
        There may be no such recent major merger event satisfying the
        criteria, since we only tracing ``tau_merger`` (~3 Gyr) back.
        On the other hand, the cluster may only experience minor merger
        or accretion events.

        Catalog columns
        ---------------
        * ``rmm_mass1``, ``rmm_mass2`` : masses of the main and sub
          clusters upon the recent major merger event; unit: [Msun]
        * ``rmm_z``, ``rmm_age`` : redshift and cosmic age; unit: [Gyr]
          of the recent major merger event.
        """
        logger.info("Simulating the galaxy formation to identify " +
                    "the most recent major merger event ...")
        num = len(self.catalog)
        mdata = np.zeros(shape=(num, 4))
        mdata.fill(np.nan)
        num_major = 0  # number of clusters with recent major merger

        for i, row in zip(range(num), self.catalog.itertuples()):
            ii = i + 1
            if ii % 50 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            z0, M0 = row.z, row.mass
            age0 = cosmo.age(z0)
            zmax = cosmo.redshift(age0 - self.tau_merger)
            clform = ClusterFormation(M0=M0, z0=z0, zmax=zmax,
                                      ratio_major=self.ratio_major,
                                      merger_mass_min=self.merger_mass_min)
            clform.simulate_mergertree(main_only=True)
            mmev = clform.recent_major_merger
            if mmev:
                num_major += 1
                mdata[i, :] = [mmev["M_main"], mmev["M_sub"],
                               mmev["z"], mmev["age"]]

        mdf = pd.DataFrame(data=mdata,
                           columns=["rmm_mass1", "rmm_mass2",
                                    "rmm_z", "rmm_age"])
        self.catalog = self.catalog.join(mdf, how="outer")
        self.catalog_comment += [
            "rmm_mass1 : main cluster mass at recent major merger; [Msun]",
            "rmm_mass2 : sub cluster mass at recent major merger; [Msun]",
            "rmm_z : redshift of the recent major merger",
            "rmm_age : cosmic age of the recent major merger; [Gyr]",
        ]
        logger.info("Simulated and identified recent major merger events.")
        logger.info("%d (%.1f%%) clusters have recent major mergers." %
                    (num_major, 100*num_major/num))

        """
        """

    def preprocess(self):
        """
        Perform the preparation procedures for the later simulations.

        Attributes
        ----------
        _preprocessed : bool
            This attribute presents and is ``True`` after the preparation
            procedures have been done.
        """
        if hasattr(self, "_preprocessed") and self._preprocessed:
            return

        logger.info("{name}: preprocessing ...".format(name=self.name))
        self._load_catalog()
        self._process_catalog()
        self._simulate_mergers()

        # TODO ???

        self._preprocessed = True

    def postprocess(self):
        """
        Do some necessary post-simulation operations.
        """
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the final resulting clusters catalog
        logger.info("Save the resulting catalog ...")
        if self.catalog_outfile is None:
            logger.warning("Catalog output file not set; skip saving!")
        else:
            dataframe_to_csv(self.catalog, outfile=self.catalog_outfile,
                             comment=self.catalog_comment,
                             clobber=self.clobber)
