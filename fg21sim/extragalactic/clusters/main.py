# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate the extended radio emissions from galaxy clusters due to
merger-induced turbulence and/or shock accelerations,
e.g., (giant) radio halos, (elongated double) radio relics.

NOTE
----
There are other types of extended radio emissions not considered
yet, e.g., mini-halos, roundish radio relics, etc.
"""

import os
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from .psformalism import PSFormalism
from .formation import ClusterFormation
from .halo import RadioHalo
from ...share import CONFIGS, COSMO
from ...utils.io import (dataframe_to_csv, csv_to_dataframe,
                         pickle_dump, pickle_load)
from ...utils.ds import dictlist_to_dataframe
from ...utils.convert import JyPerPix_to_K
from ...sky import get_sky
from . import helper


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
    sky : `~SkyBase`
        The sky instance to deal with the simulation sky as well as the
        output map.
        XXX: current full-sky HEALPix map is NOT supported!
    """
    # Component name
    compID = "extragalactic/clusters"
    name = "galaxy clusters (halos)"

    def __init__(self, configs=CONFIGS):
        self.configs = configs
        self._set_configs()

        self.sky = get_sky(configs)
        self.sky.add_header("CompID", self.compID, "Emission component ID")
        self.sky.add_header("CompName", self.name, "Emission component")
        self.sky.add_header("BUNIT", "K", "[Kelvin] Data unit")
        self.sky.creator = __name__

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = self.compID
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.use_output_catalog = self.configs.getn(comp+"/use_output_catalog")
        self.halos_catalog_outfile = self.configs.get_path(
            comp+"/halos_catalog_outfile")
        self.halos_data_dumpfile = os.path.splitext(
            self.halos_catalog_outfile)[0] + ".pkl"
        self.dump_halos_data = self.configs.getn(comp+"/dump_halos_data")
        self.use_dump_halos_data = self.configs.getn(
            comp+"/use_dump_halos_data")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        self.merger_mass_min = self.configs.getn(comp+"/merger_mass_min")
        self.ratio_major = self.configs.getn(comp+"/ratio_major")
        self.use_max_merger = self.configs.getn(comp+"/use_max_merger")
        self.tau_merger = self.configs.getn(comp+"/tau_merger")
        self.frequencies = self.configs.frequencies
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.clobber = self.configs.getn("output/clobber")
        logger.info("Loaded and set up configurations")

        if self.use_dump_halos_data and (not self.use_output_catalog):
            self.use_output_catalog = True
            logger.warning("Forced to use existing cluster catalog, "
                           "due to 'use_dump_halos_data=True'")

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
        psform.calc_dndlnm()
        psform.write()
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
                       elliptical semi-major axis to semi-minor axis
        * ``rotation`` : rotation angle; uniformly distributed within
                         [0, 360.0); unit: [deg]

        NOTE
        ----
        felong (elongated fraction) ::
            Adopt a definition (felong = b/a) similar to the Hubble
            classification for the elliptical galaxies.  Considering that
            radio halos are generally regular, ``felong`` is thus restricted
            within [0.6, 1.0], and sampled from a cut and absolute normal
            distribution centered at 1.0 with sigma ~0.4/3 (<= 3σ).
        """
        logger.info("Preliminary processes to the catalog ...")
        num = len(self.catalog)
        lon, lat = self.sky.random_points(n=num)
        self.catalog["lon"] = lon
        self.catalog["lat"] = lat
        self.catalog_comment.append(
            "lon, lat : longitudes and latitudes [deg]")
        logger.info("Added catalog columns: lon, lat.")

        felong_min = 0.6
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
        Simulate the *recent major/maximum merger* event for each cluster.

        First simulate the cluster formation history by tracing the
        merger and accretion events of the main cluster, then identify
        the most recent major merger event according to the mass ratio
        of two merging clusters.  If ``self.use_max_merger=True`` then
        the recent maximum merger (associated with biggest sub cluster)
        is selected.  The merger event properties are then appended to
        the catalog for subsequent radio halo simulation.

        NOTE
        ----
        There may be no such recent *major* merger event satisfying the
        criteria, since we only tracing ``tau_merger`` (~2-3 Gyr) back.
        On the other hand, the cluster may only experience minor merger
        or accretion events.

        Catalog columns
        ---------------
        * ``rmm_mass1``, ``rmm_mass2`` : [Msun] masses of the main and sub
          clusters upon the recent major/maximum merger event;
        * ``rmm_z``, ``rmm_age`` : redshift and cosmic age [Gyr]
          of the recent major/maximum merger event.
        """
        logger.info("Simulating the galaxy formation to identify " +
                    "the most recent major/maximum merger event ...")
        if self.use_max_merger:
            logger.info("Use the recent *maximum* merger event!")
        else:
            logger.info("Use the recent *major* merger event!")
        num = len(self.catalog)
        mdata = np.zeros(shape=(num, 4))
        mdata.fill(np.nan)

        for i, row in zip(range(num), self.catalog.itertuples()):
            ii = i + 1
            if ii % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            z0, M0 = row.z, row.mass
            age0 = COSMO.age(z0)
            zmax = COSMO.redshift(age0 - self.tau_merger)
            clform = ClusterFormation(M0=M0, z0=z0, zmax=zmax,
                                      ratio_major=self.ratio_major,
                                      merger_mass_min=self.merger_mass_min)
            clform.simulate_mergertree(main_only=True)
            if self.use_max_merger:
                # NOTE: may be ``None`` due to no mergers occurred at all!
                mmev = clform.max_merger
            else:
                mmev = clform.recent_major_merger
            if mmev:
                mdata[i, :] = [mmev["M_main"], mmev["M_sub"],
                               mmev["z"], mmev["age"]]

        mdf = pd.DataFrame(data=mdata,
                           columns=["rmm_mass1", "rmm_mass2",
                                    "rmm_z", "rmm_age"])
        self.catalog = self.catalog.join(mdf, how="outer")
        self.catalog_comment += [
            "rmm_mass1 : [Msun] main cluster mass of recent major/maximum merger",
            "rmm_mass2 : [Msun] sub cluster mass of recent major/maximum merger",
            "rmm_z : redshift of the recent major/maximum merger",
            "rmm_age : [Gyr] cosmic age at the recent major/maximum merger",
        ]
        logger.info("Simulated and identified recent major/maximum mergers.")
        if not self.use_max_merger:
            num_major = np.sum(~mdf["rmm_z"].isnull())
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
            data = OrderedDict([
                ("z0", halo.z_obs),
                ("M0", halo.M_obs),  # [Msun]
                ("Rvir0", halo.radius_virial_obs),  # [kpc]
                ("kT0", halo.kT_obs),  # [keV]
                ("B0", halo.B_obs),  # [uG] magnetic field at z_obs
                ("lon", row.lon),  # [deg] longitude
                ("lat", row.lat),  # [deg] longitude
                ("felong", row.felong),  # Fraction of elongation
                ("rotation", row.rotation),  # [deg] ellipse rotation angle
                ("M_main", halo.M_main),  # [Msun]
                ("M_sub", halo.M_sub),  # [Msun]
                ("z_merger", halo.z_merger),
                ("kT_main", halo.kT_main),  # [keV] main cluster kT at z_merger
                ("kT_sub", halo.kT_sub),  # [keV] sub-cluster kT at z_merger
                ("Rvir_main", halo.radius_virial_main),  # [kpc] at z_merger
                ("Rvir_sub", halo.radius_virial_sub),  # [kpc] at z_merger
                ("tback_merger", halo.tback_merger),  # [Gyr]
                ("time_turbulence", halo.time_turbulence),  # [Gyr]
                ("Rhalo", halo.radius),  # [kpc]
                ("Rhalo_angular", halo.angular_radius),  # [arcsec]
                ("volume", halo.volume),  # [kpc^3]
                ("Mach_turb", halo.Mach_turbulence),  # turbulence Mach number
                ("tau_acc", halo.tau_acceleration),  # [Gyr]
                ("Ke", halo.injection_rate),  # [cm^-3 Gyr^-1]
                ("gamma", halo.gamma),  # Lorentz factors
                ("n_e", n_e),  # [cm^-3]
            ])
            self.halos.append(data)
        logger.info("Simulated radio halos for merging cluster.")

    def _calc_halos_emission(self):
        """
        Calculate the radio emissions at configured frequencies.
        """
        logger.info("Calculating the radio emissions for halos ...")
        num = len(self.halos)
        i = 0
        for hdict in self.halos:
            i += 1
            if i % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (i, num, 100*i/num))

            halo = RadioHalo(M_obs=hdict["M0"], z_obs=hdict["z0"],
                             M_main=hdict["M_main"], M_sub=hdict["M_sub"],
                             z_merger=hdict["z_merger"],
                             configs=self.configs)
            halo.set_electron_spectrum(hdict["n_e"])

            emissivity = halo.calc_emissivity(frequencies=self.frequencies)
            power = halo.calc_power(self.frequencies, emissivity=emissivity)
            # k-correction considered
            flux = halo.calc_flux(self.frequencies)
            Tb_mean = halo.calc_brightness_mean(self.frequencies, flux=flux,
                                                pixelsize=self.sky.pixelsize)
            # Update or add new items
            hdict["frequency"] = self.frequencies  # [MHz]
            hdict["emissivity"] = emissivity  # [erg/s/cm^3/Hz]
            hdict["power"] = power  # [W/Hz]
            hdict["flux"] = flux  # [Jy]
            hdict["Tb_mean"] = Tb_mean  # [K]
        logger.info("Done calculate the radio emissions.")

    def _draw_halos(self):
        """
        Draw the template images for each halo, and cache them for
        simulating the superimposed halos at requested frequencies.

        NOTE
        ----
        The drawn template images are append to the dictionaries of
        the corresponding halo within the ``self.halos``.
        The templates are normalized to have *mean* value of 1.
        """
        num = len(self.halos)
        logger.info("Draw template images for %d halos ..." % num)
        i = 0
        for hdict in self.halos:
            i += 1
            if i % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (i, num, 100*i/num))
            theta_e = hdict["Rhalo_angular"] / self.sky.pixelsize
            template = helper.draw_halo(radius=theta_e,
                                        felong=hdict["felong"],
                                        rotation=hdict["rotation"])
            hdict["template"] = template
        logger.info("Done drawn halo template images.")

    def _save_halos_catalog(self, outfile=None):
        """
        Convert the halos data (``self.halos``) into a Pandas DataFrame
        and write into a CSV file.
        """
        if outfile is None:
            outfile = self.halos_catalog_outfile

        logger.info("Converting halos data to be a Pandas DataFrame ...")
        keys = list(self.halos[0].keys())
        # Ignore the ``gamma`` and ``n_e`` items
        for k in ["gamma", "n_e", "template"]:
            keys.remove(k)
        halos_df = dictlist_to_dataframe(self.halos, keys=keys)
        dataframe_to_csv(halos_df, outfile, clobber=self.clobber)
        logger.info("Saved DataFrame of halos data to file: %s" % outfile)

    def _dump_halos_data(self, outfile=None):
        """
        Dump the simulated halos data into Python native pickle format,
        making it possible to load the data back to quickly calculate
        the emissions at additional frequencies.
        """
        if outfile is None:
            outfile = self.halos_data_dumpfile
        pickle_dump(self.halos, outfile=outfile, clobber=self.clobber)

    def _outfilepath(self, frequency, **kwargs):
        """
        Generate the path/filename to the output file for writing
        the simulate sky images.

        Parameters
        ----------
        frequency : float
            The frequency of the output sky image.
            Unit: [MHz]

        Returns
        -------
        filepath : str
            The generated filepath for the output sky file.
        """
        filename = self.filename_pattern.format(
            prefix=self.prefix, frequency=frequency, **kwargs)
        filepath = os.path.join(self.output_dir, filename)
        return filepath

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
        if self.use_output_catalog:
            logger.info("Use existing cluster & halo catalog: %s" %
                        self.catalog_outfile)
            self.catalog, self.catalog_comment = csv_to_dataframe(
                self.catalog_outfile)
            ncluster = len(self.catalog)
            idx_rmm = ~self.catalog["rmm_z"].isnull()
            nhalo = idx_rmm.sum()
            logger.info("Loaded cluster catalog: %d clusters with %d halos" %
                        (ncluster, nhalo))
        else:
            self._simulate_catalog()
            self._process_catalog()
            self._simulate_mergers()

        if self.use_dump_halos_data:
            logger.info("Use existing dumped halos raw data: %s" %
                        self.halos_data_dumpfile)
            self.halos = pickle_load(self.halos_data_dumpfile)
            logger.info("Loaded data of %d halos" % len(self.halos))
        else:
            self._simulate_halos()

        self._calc_halos_emission()
        self._draw_halos()

        self._preprocessed = True

    def simulate_frequency(self, freqidx):
        """
        Simulate the superimposed radio halos image at frequency (by
        frequency index) based on the above simulated halo templates.

        Parameters
        ----------
        freqidx : int
            The index of the frequency in the ``self.frequencies`` where
            to simulate the radio halos image.

        Returns
        -------
        sky : `~SkyBase`
            The simulated sky image of radio halos as a new sky instance.
        """
        freq = self.frequencies[freqidx]
        logger.info("Simulating radio halo map at %.2f [MHz] ..." % freq)
        sky = self.sky.copy()
        sky.frequency = freq
        # Conversion factor for [Jy/pixel] to [K]
        JyPP2K = JyPerPix_to_K(freq, sky.pixelsize)

        for hdict in self.halos:
            center = (hdict["lon"], hdict["lat"])
            template = hdict["template"]  # normalized to have mean of 1
            Npix = template.size
            flux = hdict["flux"][freqidx]  # [Jy]
            Tmean = (flux/Npix) * JyPP2K  # [K]
            Timg = Tmean * template  # [K]
            sky.add(Timg, center=center)

        logger.info("Done simulate map at %.2f [MHz]." % freq)
        return sky

    def simulate(self):
        """
        Simulate the sky images of radio halos at each frequency.

        Returns
        -------
        skyfiles : list[str]
            List of the filepath to the written sky files
        """
        logger.info("Simulating {name} ...".format(name=self.name))
        skyfiles = []
        for idx, freq in enumerate(self.frequencies):
            sky = self.simulate_frequency(freqidx=idx)
            outfile = self._outfilepath(frequency=freq)
            sky.write(outfile)
            skyfiles.append(outfile)
        logger.info("Done simulate {name}!".format(name=self.name))
        return skyfiles

    def postprocess(self):
        """
        Do some necessary post-simulation operations.
        """
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the final resulting clusters catalog
        logger.info("Save the resulting catalog ...")
        if self.use_output_catalog:
            logger.info("No need to save the cluster catalog.")
        else:
            dataframe_to_csv(self.catalog, outfile=self.catalog_outfile,
                             comment=self.catalog_comment,
                             clobber=self.clobber)

        # Save the simulated halos catalog and raw data
        logger.info("Saving the simulated halos catalog and raw data ...")
        if self.use_dump_halos_data:
            filepath = self.halos_catalog_outfile
            os.rename(filepath, filepath+".old")
            logger.warning("Backed up halos catalog: %s -> %s" %
                           (filepath, filepath+".old"))
            filepath = self.halos_data_dumpfile
            os.rename(filepath, filepath+".old")
            logger.warning("Backed up halos data dump file: %s -> %s" %
                           (filepath, filepath+".old"))
        self._save_halos_catalog()
        if self.dump_halos_data:
            self._dump_halos_data()
