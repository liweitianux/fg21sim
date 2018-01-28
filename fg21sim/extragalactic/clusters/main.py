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
from .halo import RadioHaloAM
from .emission import HaloEmission
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
        self._set_configs(configs)
        self.sky = get_sky(configs)
        self.sky.add_header("CompID", self.compID, "Emission component ID")
        self.sky.add_header("CompName", self.name, "Emission component")
        self.sky.add_header("BUNIT", "K", "[Kelvin] Data unit")
        self.sky.creator = __name__

    def _set_configs(self, configs):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = self.compID
        self.configs = configs
        self.catalog_outfile = configs.get_path(comp+"/catalog_outfile")
        self.dump_catalog_data = configs.getn(comp+"/dump_catalog_data")
        self.use_dump_catalog_data = configs.getn(
            comp+"/use_dump_catalog_data")
        self.halos_catalog_outfile = configs.get_path(
            comp+"/halos_catalog_outfile")
        self.dump_halos_data = configs.getn(comp+"/dump_halos_data")
        self.use_dump_halos_data = configs.getn(
            comp+"/use_dump_halos_data")
        self.halo_dropout = configs.getn(comp+"/halo_dropout")
        self.prefix = configs.getn(comp+"/prefix")
        self.output_dir = configs.get_path(comp+"/output_dir")
        self.merger_mass_min = configs.getn(comp+"/merger_mass_min")
        self.time_traceback = configs.getn(comp+"/time_traceback")
        self.frequencies = configs.frequencies
        self.filename_pattern = configs.getn("output/filename_pattern")
        self.clobber = configs.getn("output/clobber")
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

        * Calculate the cosmic age at cluster's redshift
        * Generate random positions within the sky for each cluster;
        * Generate random elongated fraction;
        * Generate random rotation angle.

        Catalog Items
        -------------
        age : [Gyr] cosmic age at cluster's redshift, ~ cluster age
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
                ("age", COSMO.age(cdict["z"])),
                ("lon", lon[i]),
                ("lat", lat[i]),
                ("felong", felong[i]),
                ("rotation", rotation[i]),
            ])
        self.comments += [
            "age - [Gyr] cosmic age at z; ~ cluster age",
            "lon, lat - [deg] longitudes and latitudes",
            "felong - elongated fraction (= b/a)",
            "rotation -  [deg] ellipse rotation angle",
        ]
        logger.info("Added catalog items: age, lon, lat, felong, rotation.")

    def _simulate_mergers(self):
        """
        Simulate the formation history of each cluster to build their
        merger histories.

        Catalog Items
        -------------
        merger_num : number of merger events within the traced period
        merger_mass1, merger_mass2 :
            [Msun] masses of the main and sub clusters of each merger.
        merger_z, merger_age : redshifts and cosmic ages [Gyr]
            of each merger event, in backward time ordering.
        """
        logger.info("Simulating merger histories for each cluster ...")
        num = len(self.catalog)
        num_hasmerger = 0
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
            mergers = clform.mergers()
            if mergers:
                num_hasmerger += 1
                cdict.update([
                    ("merger_num",   len(mergers)),
                    ("merger_mass1", [ev["M_main"] for ev in mergers]),
                    ("merger_mass2", [ev["M_sub"] for ev in mergers]),
                    ("merger_z",     [ev["z"] for ev in mergers]),
                    ("merger_age",   [ev["age"] for ev in mergers]),
                ])
            else:
                cdict.update([
                    ("merger_num",   0),
                    ("merger_mass1", []),
                    ("merger_mass2", []),
                    ("merger_z",     []),
                    ("merger_age",   []),
                ])

        self.comments += [
            "merger_num - number of merger events",
            "merger_mass1 - [Msun] main cluster mass of each merger",
            "merger_mass2 - [Msun] sub cluster mass of each merger",
            "merger_z - redshift of each merger",
            "merger_age - [Gyr] cosmic age at each merger",
        ]
        logger.info("%d (%.1f%%) clusters experienced recent mergers." %
                    (num_hasmerger, 100*num_hasmerger/num))
        nmax = max([cdict["merger_num"] for cdict in self.catalog])
        logger.info("Maximum number of merger events: %d" % nmax)

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
                   if cdict["merger_num"] > 0]
        num = len(idx_rmm)
        logger.info("Simulating halos for %d clusters with mergers ..." % num)
        self.halos = []
        for i, idx in enumerate(idx_rmm):
            ii = i + 1
            if ii % 50 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            cdict = self.catalog[idx]
            merger_num = cdict["merger_num"]
            M_obs = cdict["mass"]
            z_obs = cdict["z"]
            M1 = cdict["merger_mass1"][merger_num-1]
            z1 = cdict["merger_z"][merger_num-1]
            info = ("[%d/%d] " % (ii, num) +
                    "M(%.2e)@z(%.3f) -> M(%.2e)@z(%.3f) with %d merger(s)" %
                    (M1, z1, M_obs, z_obs, merger_num))
            logger.info(info)
            halo = RadioHaloAM(M_obs=M_obs, z_obs=z_obs,
                               M_main=cdict["merger_mass1"],
                               M_sub=cdict["merger_mass2"],
                               z_merger=cdict["merger_z"],
                               merger_num=merger_num,
                               configs=self.configs)
            n_e = halo.calc_electron_spectrum()
            data = OrderedDict([
                ("z0", z_obs),
                ("M0", M_obs),  # [Msun]
                ("age0", halo.age_obs),  # [Gyr]
                ("merger_num", merger_num),
                ("lon", cdict["lon"]),  # [deg] longitude
                ("lat", cdict["lat"]),  # [deg] longitude
                ("felong", cdict["felong"]),  # fraction of elongation
                ("rotation", cdict["rotation"]),  # [deg] rotation angle
                ("Rvir0", halo.radius_virial_obs),  # [kpc]
                ("kT0", halo.kT_obs),  # [keV]
                ("B0", halo.B_obs),  # [uG] magnetic field @ z_obs
                ("Rhalo", halo.radius),  # [kpc]
                ("Rhalo_angular", halo.angular_radius),  # [arcsec]
                ("volume", halo.volume),  # [kpc^3]
                ("Ke", halo.injection_rate),  # [cm^-3 Gyr^-1]
                ("time_turbulence", halo.time_turbulence_avg),  # [Gyr]
                ("Mach_turb", halo.mach_turbulence_avg),  # Mach number
                ("tau_acc", halo.tau_acceleration_avg),  # [Gyr]
                ("tfrac_acc", halo.time_acceleration_fraction),
                ("gamma", halo.gamma),  # Lorentz factors
                ("n_e", n_e),  # [cm^-3]
            ])
            self.halos.append(data)
        logger.info("Simulated radio halos for clusters with recent mergers.")

    def _calc_halos_emission(self):
        """
        Calculate the radio emissions at configured frequencies.
        """
        logger.info("Calculating the radio halo emissions ...")
        num = len(self.halos)
        for i, hdict in enumerate(self.halos):
            ii = i + 1
            if ii % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
            haloem = HaloEmission(gamma=hdict["gamma"], n_e=hdict["n_e"],
                                  B=hdict["B0"], radius=hdict["Rhalo"],
                                  redshift=hdict["z0"])
            emissivity = haloem.calc_emissivity(frequencies=self.frequencies)
            power = haloem.calc_power(self.frequencies, emissivity=emissivity)
            flux = haloem.calc_flux(self.frequencies)
            Tb_mean = haloem.calc_brightness_mean(self.frequencies, flux=flux,
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

        If ``halo_dropout`` is given, then select the specified number of
        most powerful radio halos from the catalog, and mark them with
        a property ``drop=True``, which will then be excluded from the
        following halo drawing step, in order to obtain more stable
        simulation results.

        NOTE
        ----
        If the halo data is reloaded from a previously dumped catalog,
        the original dropout markers is just ignored.
        """
        if self.halo_dropout <= 0:
            logger.info("No need to drop out halos.")
            return

        power = np.array([hdict["power"][0] for hdict in self.halos])
        argsort = power.argsort()[::-1]  # max -> min
        idx_drop = argsort[:self.halo_dropout]
        num = 0
        for i, hdict in enumerate(self.halos):
            if i in idx_drop:
                hdict["drop"] = True
            else:
                hdict["drop"] = False
                num += 1
        logger.info("Marked %d most powerful halos for dropping" %
                    self.halo_dropout)
        logger.info("Remaining number of halos: %d" % num)

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
        idx_kept = [idx for idx, cdict in enumerate(self.halos)
                    if not cdict.get("drop", False)]
        num = len(idx_kept)
        logger.info("Draw template images for %d halos ..." % num)
        for i, idx in enumerate(idx_kept):
            hdict = self.halos[idx]
            ii = i + 1
            if ii % 100 == 0:
                logger.info("[%d/%d] %.1f%% ..." % (ii, num, 100*ii/num))
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
        # Pad the merger events to be same length
        nmax = max([cdict["merger_num"] for cdict in self.catalog])
        for cdict in self.catalog:
            num = len(cdict["merger_z"])
            if num == nmax:
                continue
            cdict.update([
                ("merger_mass1",
                 list(cdict["merger_mass1"]) + [None]*(nmax-num)),
                ("merger_mass2",
                 list(cdict["merger_mass2"]) + [None]*(nmax-num)),
                ("merger_z",
                 list(cdict["merger_z"]) + [None]*(nmax-num)),
                ("merger_age",
                 list(cdict["merger_age"]) + [None]*(nmax-num)),
            ])

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
