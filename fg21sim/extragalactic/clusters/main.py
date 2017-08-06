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

import os
import logging

import numpy as np
import pandas as pd

from .psformalism import PSFormalism
from .formation import ClusterFormation
from .halo import RadioHalo
from ...share import CONFIGS, COSMO
from ...utils.io import dataframe_to_csv, pickle_dump
from ...utils.ds import dictlist_to_dataframe
from ...sky import get_sky


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

    def __init__(self, configs=CONFIGS):
        self.configs = configs
        self.sky = get_sky(configs)
        self._set_configs()

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = "extragalactic/clusters"
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.halos_dumpfile = self.configs.get_path(comp+"/halos_dumpfile")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        self.merger_mass_min = self.configs.getn(comp+"/merger_mass_min")
        self.ratio_major = self.configs.getn(comp+"/ratio_major")
        self.tau_merger = self.configs.getn(comp+"/tau_merger")

        self.frequencies = self.configs.frequencies
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

    def _simulate_catalog(self):
        """
        Simulate the (z, mass) catalog of the cluster distribution
        according to the Press-Schechter formalism.

        Catalog columns
        ---------------
        * ``z`` : redshifts
        * ``mass`` : cluster total mass; unit: [Msun]
        """
        logger.info("Simulating the clusters (z, mass) catalog ...")
        psform = PSFormalism(configs=self.configs)
        counts = psform.calc_cluster_counts(coverage=self.sky.area)
        self.catalog, self.catalog_comment = psform.sample_z_m(counts)
        logger.info("Simulated cluster catalog of counts %d." % counts)

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
            age0 = COSMO.age(z0)
            zmax = COSMO.redshift(age0 - self.tau_merger)
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

    def _simulate_halos(self):
        """
        Simulate the radio halo properties for each cluster with
        recent major merger event.

        Attributes
        ----------
        halos : list[dict]
            Simulated data for each cluster with recent major merger.
        halos_df : `~pandas.DataFrame`
            The Pandas DataFrame converted from the above ``halos`` data.
        """
        # Select out the clusters with recent major mergers
        idx_rmm = ~self.catalog["rmm_z"].isnull()
        num = idx_rmm.sum()
        logger.info("Simulating halos for %d merging clusters ..." % num)
        self.halos = []
        i = 0
        for row in self.catalog[idx_rmm].itertuples():
            i += 1
            if i % 50 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (i, num, 100*i/num))
            logger.info("[%d/%d] " % (i, num) +
                        "M1[%.2e] & M2[%.2e] @ z[%.3f] -> M[%.2e] @ z[%.3f]" %
                        (row.rmm_mass1, row.rmm_mass2, row.rmm_z,
                         row.mass, row.z))
            halo = RadioHalo(M_obs=row.mass, z_obs=row.z,
                             M_main=row.rmm_mass1, M_sub=row.rmm_mass2,
                             z_merger=row.rmm_z, configs=self.configs)
            n_e = halo.calc_electron_spectrum()
            emissivity = halo.calc_emissivity(frequencies=self.frequencies)
            power = halo.calc_power(emissivity)
            flux = halo.calc_flux(emissivity)
            Tb_mean = halo.calc_brightness_mean(emissivity, self.frequencies,
                                                pixelsize=self.sky.pixelsize)
            data = {
                "z0": halo.z_obs,
                "M0": halo.M_obs,  # [Msun]
                "z_merger": halo.z_merger,
                "M_main": halo.M_main,  # [Msun]
                "M_sub": halo.M_sub,  # [Msun]
                "time_crossing": halo.time_crossing,  # [Gyr]
                "gamma": halo.gamma,  # Lorentz factors
                "radius": halo.radius,  # [kpc]
                "angular_radius": halo.angular_radius,  # [arcsec]
                "volume": halo.volume,  # [cm^3]
                "B": halo.magnetic_field,  # [uG]
                "n_e": n_e,  # [cm^-3]
                "frequencies": self.frequencies,  # [MHz]
                "emissivity": emissivity,  # [erg/s/cm^3/Hz]
                "power": power,  # [W/Hz]
                "flux": flux,  # [Jy]
                "Tb_mean": Tb_mean,  # [K]
            }
            self.halos.append(data)
        logger.info("Simulated radio halos for merging cluster.")
        #
        logger.info("Converting halos data to be a Pandas DataFrame ...")
        # Ignore the ``gamma`` and ``n_e`` items
        keys = ["z0", "M0", "z_merger", "M_main", "M_sub",
                "time_crossing", "radius", "angular_radius",
                "B", "frequencies", "emissivity", "flux", "Tb_mean"]
        self.halos_df = dictlist_to_dataframe(self.halos, keys=keys)
        logger.info("Done halos data conversion.")

    def _draw_halos(self):
        pass

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
        self._simulate_catalog()
        self._process_catalog()
        self._simulate_mergers()
        self._simulate_halos()
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
        # Dump the simulated clusters data
        logger.info("Dumping the simulated halos data ...")
        if self.halos_dumpfile is None:
            logger.warning("Missing dump outfile; skip dump cluster data!")
        else:
            pickle_dump(self.halos, outfile=self.halos_dumpfile,
                        clobber=self.clobber)
        # Also save converted DataFrame of halos data
        outfile = os.path.splitext(self.halos_dumpfile)[0] + ".csv"
        dataframe_to_csv(self.halos_df, outfile, clobber=self.clobber)
        logger.info("Saved DataFrame of halos data to file: %s" % outfile)
