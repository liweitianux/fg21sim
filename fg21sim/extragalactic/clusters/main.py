# Copyright (c) 2017-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate the diffuse radio emissions from galaxy clusters due to
merger-induced turbulence and/or shock accelerations, e.g.,
(giant) radio halos, (elongated double) radio relics.

NOTE
----
There are other types of diffuse radio emissions not considered
yet, e.g., mini-halos, roundish radio relics.
"""

import os
import logging
from collections import OrderedDict

import numpy as np

from .psformalism import PSFormalism
from .formation import ClusterFormation
from .halo import RadioHalo
from ...share import CONFIGS, COSMO
from ...utils.io import dataframe_to_csv, pickle_dump, pickle_load
from ...utils.ds import dictlist_to_dataframe
from ...utils.convert import JyPerPix_to_K
from ...sky import get_sky
from . import helper


logger = logging.getLogger(__name__)


class GalaxyClusters:
    """
    Simulate the diffuse radio emissions from the galaxy clusters.

    NOTE
    ----
    Currently only implement the *giant radio halos*, while other types
    of diffuse emissions are missing, e.g., mini-halos, elongated relics,
    roundish relics.

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
        self.dump_catalog_data = self.configs.getn(comp+"/dump_catalog_data")
        self.use_dump_catalog_data = self.configs.getn(
            comp+"/use_dump_catalog_data")
        self.halos_catalog_outfile = self.configs.get_path(
            comp+"/halos_catalog_outfile")
        self.dump_halos_data = self.configs.getn(comp+"/dump_halos_data")
        self.use_dump_halos_data = self.configs.getn(
            comp+"/use_dump_halos_data")
        self.halo_dropout = self.configs.getn(comp+"/halo_dropout")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        self.merger_mass_min = self.configs.getn(comp+"/merger_mass_min")
        self.ratio_major = self.configs.getn(comp+"/ratio_major")
        self.use_max_merger = self.configs.getn(comp+"/use_max_merger")
        self.time_traceback = self.configs.getn(comp+"/time_traceback")
        self.frequencies = self.configs.frequencies
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.clobber = self.configs.getn("output/clobber")
        logger.info("Loaded and set up configurations")

        if self.use_dump_halos_data and (not self.use_dump_catalog_data):
            self.use_dump_catalog_data = True
            logger.warning("Forced to use existing cluster catalog, "
                           "due to 'use_dump_halos_data=True'")

    def _simulate_catalog(self):
        """
        Simulate the (z, mass) catalog of the cluster distribution
        according to the Press-Schechter formalism.

        Catalog Items
        -------------
        z : redshifts
        mass : [Msun] cluster total mass

        Attributes
        ----------
        catalog : list[dict]
        comments : list[str]
        """
        logger.info("Simulating the clusters (z, mass) catalog ...")
        psform = PSFormalism(configs=self.configs)
        psform.calc_dndlnm()
        psform.write()
        counts = psform.calc_cluster_counts(coverage=self.sky.area)
        z, mass, self.comments = psform.sample_z_m(counts)
        self.catalog = []
        for z_, m_ in zip(z, mass):
            self.catalog.append(OrderedDict([("z", z_), ("mass", m_)]))
        logger.info("Simulated a catalog of %d clusters" % counts)

    def _process_catalog(self):
        """
        Do some basic processes to the catalog:

        * Generate random positions within the sky for each cluster;
        * Generate random elongated fraction;
        * Generate random rotation angle.

        Catalog Items
        -------------
        lon : [deg] longitudes
        lat : [deg] latitudes
        felong : elongated fraction, defined as the ratio of
                 elliptical semi-major axis to semi-minor axis
        rotation : [deg] rotation angle; uniformly distributed within
                   [0, 360.0)

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
        felong_min = 0.6
        sigma = (1.0 - felong_min) / 3.0
        felong = 1.0 - np.abs(np.random.normal(scale=sigma, size=num))
        felong[felong < felong_min] = felong_min
        rotation = np.random.uniform(low=0.0, high=360.0, size=num)

        for i, cdict in enumerate(self.catalog):
            cdict.update([
                ("lon", lon[i]),
                ("lat", lat[i]),
                ("felong", felong[i]),
                ("rotation", rotation[i]),
            ])
        self.comments += [
            "lon, lat - [deg] longitudes and latitudes",
            "felong - elongated fraction (= b/a)",
            "rotation -  [deg] ellipse rotation angle",
        ]
        logger.info("Added catalog items: lon, lat, felong, rotation.")

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
        criteria, since we only trace ``time_traceback`` (~2-3 Gyr) back.
        On the other hand, the cluster may only experience minor merger
        or accretion events.

        Catalog Items
        -------------
        rmm_mass1, rmm_mass2 :
            [Msun] masses of the main and sub clusters upon the recent
            major/maximum merger event
        rmm_z, rmm_age : redshift and cosmic age [Gyr]
          of the recent major/maximum merger event.
        """
        logger.info("Simulating the galaxy formation to identify " +
                    "the most recent major/maximum merger event ...")
        if self.use_max_merger:
            logger.info("Use the recent *maximum* merger event!")
        else:
            logger.info("Use the recent *major* merger event!")

        num = len(self.catalog)
        num_rmm = 0
        for i, cdict in enumerate(self.catalog):
            ii = i + 1
            if ii % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            z0, M0 = cdict["z"], cdict["mass"]
            age0 = COSMO.age(z0)
            zmax = COSMO.redshift(age0 - self.time_traceback)
            clform = ClusterFormation(M0=M0, z0=z0, zmax=zmax,
                                      merger_mass_min=self.merger_mass_min)
            clform.simulate_mtree(main_only=True)
            if self.use_max_merger:
                # NOTE: may be ``{}`` due to no mergers occurred.
                mmev = clform.maximum_merger()
            else:
                mmev = clform.recent_major_merger(ratio_major=self.ratio_major)
            if mmev:
                num_rmm += 1
            cdict.update([
                ("rmm_mass1", mmev.get("M_main")),
                ("rmm_mass2", mmev.get("M_sub")),
                ("rmm_z", mmev.get("z")),
                ("rmm_age", mmev.get("age")),
            ])

        self.comments += [
            "rmm_mass1 - [Msun] main cluster mass of recent major/max merger",
            "rmm_mass2 - [Msun] sub cluster mass of recent major/max merger",
            "rmm_z - redshift of the recent major/maximum merger",
            "rmm_age - [Gyr] cosmic age at the recent major/maximum merger",
        ]
        logger.info("%d (%.1f%%) clusters have recent major/maximum mergers." %
                    (num_rmm, 100*num_rmm/num))

    def _simulate_halos(self):
        """
        Simulate the radio halo properties for each cluster with recent
        merger event.

        Attributes
        ----------
        halos : list[dict]
            Simulated data for each cluster with recent merger.
        """
        # Select out the clusters with recent mergers
        idx_rmm = [idx for idx, cdict in enumerate(self.catalog)
                   if cdict["rmm_z"] is not None]
        num = len(idx_rmm)
        logger.info("Simulating halos for %d merging clusters ..." % num)
        self.halos = []
        for i, idx in enumerate(idx_rmm):
            ii = i + 1
            if ii % 50 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            cdict = self.catalog[idx]
            z_obs = cdict["z"]
            M_obs = cdict["mass"]
            z_merger = cdict["rmm_z"]
            M_main = cdict["rmm_mass1"]
            M_sub = cdict["rmm_mass2"]
            logger.info("[%d/%d] " % (ii, num) +
                        "M1[%.2e] & M2[%.2e] @ z[%.3f] -> M[%.2e] @ z[%.3f]" %
                        (M_main, M_sub, z_merger, M_obs, z_obs))
            halo = RadioHalo(M_obs=M_obs, z_obs=z_obs,
                             M_main=M_main, M_sub=M_sub,
                             z_merger=z_merger, configs=self.configs)
            n_e = halo.calc_electron_spectrum()
            data = OrderedDict([
                ("z0", halo.z_obs),
                ("M0", halo.M_obs),  # [Msun]
                ("Rvir0", halo.radius_virial_obs),  # [kpc]
                ("kT0", halo.kT_obs),  # [keV]
                ("B0", halo.B_obs),  # [uG] magnetic field at z_obs
                ("lon", cdict["lon"]),  # [deg] longitude
                ("lat", cdict["lat"]),  # [deg] longitude
                ("felong", cdict["felong"]),  # Fraction of elongation
                ("rotation", cdict["rotation"]),  # [deg] rotation angle
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
                ("Mach_turb", halo.mach_turbulence),  # turbulence Mach number
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
        logger.info("Calculating the radio halo emissions ...")
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
            hdict.update([
                ("frequency", self.frequencies),  # [MHz]
                ("emissivity", emissivity),  # [erg/s/cm^3/Hz]
                ("power", power),  # [W/Hz]
                ("flux", flux),  # [Jy]
                ("Tb_mean", Tb_mean),  # [K]
            ])
        logger.info("Done calculate the radio emissions.")

    def _dropout_halos(self):
        """
        Considering that the (very) massive galaxy clusters are very rare,
        while the simulation sky area is rather small, therefore, once a
        very massive cluster appears, its associated radio halo is also
        very powerful and (almost) dominate other intermediate/faint halos,
        causing the simulation results unstable and have large variation.

        Drop out the specified number of most powerful radio halos from
        the catalog, in order to obtain a more stable simulation.
        """
        if self.halo_dropout <= 0:
            logger.info("No need to drop out halos.")
            return

        power = np.array([hdict["power"][0] for hdict in self.halos])
        argsort = power.argsort()[::-1]  # max -> min
        idx_drop = argsort[:self.halo_dropout]
        halos_new = [hdict for i, hdict in enumerate(self.halos)
                     if i not in idx_drop]
        self.halos = halos_new
        logger.info("Dropped out %d most powerful halos" % self.halo_dropout)
        logger.info("Remaining number of halos: %d" % len(halos_new))

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

    def _save_catalog_data(self, outfile=None, dump=None, clobber=None):
        """
        Save the simulated cluster catalog (``self.catalog``) by converting
        it into a Pandas DataFrame and writing into a CSV file.

        If ``dump=True``, then the raw data (``self.catalog``) is dumped
        into a Python pickle file, making it easier to be loaded back
        for reuse.
        """
        if outfile is None:
            outfile = self.catalog_outfile
        if dump is None:
            dump = self.dump_catalog_data
        if clobber is None:
            clobber = self.clobber

        if self.use_dump_catalog_data and os.path.exists(outfile):
            os.rename(outfile, outfile+".old")

        logger.info("Converting cluster catalog into a Pandas DataFrame ...")
        keys = list(self.catalog[0].keys())
        catalog_df = dictlist_to_dataframe(self.catalog, keys=keys)
        dataframe_to_csv(catalog_df, outfile=outfile,
                         comment=self.comments, clobber=clobber)
        logger.info("Saved cluster catalog to CSV file: %s" % outfile)

        if dump:
            outfile = os.path.splitext(outfile)[0] + ".pkl"
            if self.use_dump_catalog_data and os.path.exists(outfile):
                os.rename(outfile, outfile+".old")
            pickle_dump([self.catalog, self.comments],
                        outfile=outfile, clobber=clobber)
            logger.info("Dumped catalog raw data to file: %s" % outfile)

    def _save_halos_data(self, outfile=None, dump=None, clobber=None):
        """
        Save the simulated halo data (``self.halos``) by converting it
        into a Pandas DataFrame and writing into a CSV file.

        If ``dump=True``, then the raw data (``self.halos``) is dumped
        into a Python pickle file, making it possible to be loaded back
        to quickly calculate the emissions at additional frequencies.
        """
        if outfile is None:
            outfile = self.halos_catalog_outfile
        if dump is None:
            dump = self.dump_halos_data
        if clobber is None:
            clobber = self.clobber

        if self.use_dump_halos_data and os.path.exists(outfile):
            os.rename(outfile, outfile+".old")

        logger.info("Converting halos data into a Pandas DataFrame ...")
        keys = list(self.halos[0].keys())
        # Ignore these items: gamma, n_e, template
        for k in ["gamma", "n_e", "template"]:
            keys.remove(k)
        halos_df = dictlist_to_dataframe(self.halos, keys=keys)
        dataframe_to_csv(halos_df, outfile, clobber=clobber)
        logger.info("Saved halos data to CSV file: %s" % outfile)

        if dump:
            outfile = os.path.splitext(outfile)[0] + ".pkl"
            if self.use_dump_halos_data and os.path.exists(outfile):
                os.rename(outfile, outfile+".old")
            pickle_dump(self.halos, outfile=outfile, clobber=clobber)
            logger.info("Dumped halos raw data to file: %s" % outfile)

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
        if self.use_dump_catalog_data:
            infile = os.path.splitext(self.catalog_outfile)[0] + ".pkl"
            logger.info("Use existing cluster catalog: %s" % infile)
            self.catalog, self.comments = pickle_load(infile)
            logger.info("Loaded cluster catalog of %d clusters" %
                        len(self.catalog))
        else:
            self._simulate_catalog()
            self._process_catalog()
            self._simulate_mergers()

        if self.use_dump_halos_data:
            infile = os.path.splitext(self.halos_catalog_outfile)[0] + ".pkl"
            logger.info("Use existing halos data: %s" % infile)
            self.halos = pickle_load(infile)
            logger.info("Loaded data of %d halos" % len(self.halos))
        else:
            self._simulate_halos()

        self._calc_halos_emission()
        self._dropout_halos()
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
        logger.info("Save the cluster catalog ...")
        self._save_catalog_data()
        logger.info("Saving the simulated halos catalog and raw data ...")
        self._save_halos_data()
