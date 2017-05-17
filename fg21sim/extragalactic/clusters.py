# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Simulation of the radio emissions from clusters of galaxies.

NOTE
----
Currently, only radio *halos* are considered with many simplifications.
Radio *relics* simulations need more investigations ...
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
import astropy.units as au
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
import pandas as pd

from ..utils.fits import write_fits_healpix
from ..utils.random import spherical_uniform
from ..utils.convert import Fnu_to_Tb_fast
from ..utils.grid import make_grid_ellipse, map_grid_to_healpix


logger = logging.getLogger(__name__)


class GalaxyClusters:
    """
    Simulate the radio emissions from the clusters of galaxies, which
    host radio halos and relics (currently not considered).

    The simulation follows the method adopted by [Jelic2008]_, which uses
    the *ΛCDM deep wedge cluster catalog* derived from the *Hubble Volume
    Project* [HVP]_, [Evard2002]_.

    Every radio cluster is simulated as an *ellipse* of *uniform brightness*
    on a local coordinate grid with relatively higher resolution compared
    to the output HEALPix map, which is then mapped to the output HEALPix
    map by down-sampling, i.e., in a similar way as the simulations of SNRs.

    TODO: ???

    Parameters
    ----------
    configs : `ConfigManager`
        A `ConfigManager` instance containing default and user configurations.
        For more details, see the example configuration specifications.

    Attributes
    ----------
    TODO: ???

    NOTE
    ----
    Currently, only radio *halos* are considered with many simplifications.
    Radio *relics* simulations need more investigations ...

    References
    ----------
    .. [Jelic2008]
       Jelić, V. et al.,
       "Foreground simulations for the LOFAR-epoch of reionization experiment",
       2008, MNRAS, 389, 1319-1335,
       http://adsabs.harvard.edu/abs/2008MNRAS.389.1319J

    .. [Evard2002]
       Evard, A. E. et al.,
       "Galaxy Clusters in Hubble Volume Simulations: Cosmological Constraints
       from Sky Survey Populations",
       2002, ApJ, 573, 7-36,
       http://adsabs.harvard.edu/abs/2002ApJ...573....7E

    .. [EnBlin2002]
       Enßlin, T. A. & Röttgering, H.,
       "The radio luminosity function of cluster radio halos",
       2002, A&A, 396, 83-89,
       http://adsabs.harvard.edu/abs/2002A%26A...396...83E

    .. [Reiprich2002]
       Reiprich, Thomas H. & Böhringer, Hans,
       "The Mass Function of an X-Ray Flux-limited Sample of Galaxy Clusters",
       2002, ApJ, 567, 716-740,
       http://adsabs.harvard.edu/abs/2002ApJ...567..716R
    """
    # Component name
    name = "clusters of galaxies"

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        comp = "extragalactic/clusters"
        self.catalog_path = self.configs.get_path(comp+"/catalog")
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.halo_fraction = self.configs.getn(comp+"/halo_fraction")
        self.resolution = self.configs.getn(comp+"/resolution") * au.arcmin
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        self.nside = self.configs.getn("common/nside")
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        # Cosmology model
        self.H0 = self.configs.getn("cosmology/H0")
        self.OmegaM0 = self.configs.getn("cosmology/OmegaM0")
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.OmegaM0)
        #
        logger.info("Loaded and set up configurations")

    def _load_catalog(self):
        """Load the cluster catalog data and set up its properties."""
        self.catalog = pd.read_csv(self.catalog_path)
        nrow, ncol = self.catalog.shape
        logger.info("Loaded clusters catalog data from: {0}".format(
            self.catalog_path))
        logger.info("Clusters catalog data: {0} objects, {1} columns".format(
            nrow, ncol))
        # Set the properties for this catalog
        self.catalog_prop = {
            "omega_m": 0.3,
            "omega_lambda": 0.7,
            # Dimensionless Hubble constant
            "h": 0.7,
            "sigma8": 0.9,
            # Number of particles
            "n_particle": 1e9,
            # Particle mass of the simulation [ h^-1 ]
            "m_particle": 2.25e12 * au.solMass,
            # Cube side length [ h^-1 ]
            "l_side": 3000.0 * au.Mpc,
            # Overdensity adopted to derive the clusters
            "overdensity": 200,
            # Sky coverage
            "coverage": 10*au.deg * 10*au.deg
        }
        # Units for the catalog columns (also be populated by other methods)
        self.units = {}

    def _save_catalog_inuse(self):
        """Save the effective/inuse clusters catalog data to a CSV file."""
        if self.catalog_outfile is None:
            logger.warning("Catalog output file not set, so do NOT save.")
            return
        # Create directory if necessary
        dirname = os.path.dirname(self.catalog_outfile)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            logger.info("Created directory: {0}".format(dirname))
        # Save catalog data
        if os.path.exists(self.catalog_outfile):
            if self.clobber:
                logger.warning("Remove existing catalog file: {0}".format(
                    self.catalog_outfile))
                os.remove(self.catalog_outfile)
            else:
                raise OSError("Output file already exists: {0}".format(
                    self.catalog_outfile))
        self.catalog.to_csv(self.catalog_outfile, header=True, index=False)
        logger.info("Save clusters catalog in use to: {0}".format(
            self.catalog_outfile))

    def _process_catalog(self):
        """Process the catalog to prepare for the simulation."""
        # Dimensionless Hubble parameter adopted in THIS simulation
        h = self.H0 / 100
        logger.info("Adopted dimensionless Hubble parameter: {0}".format(h))
        # Cluster masses, unit: solMass (NOTE: h dependence)
        self.catalog["mass"] = (self.catalog["m"] *
                                self.catalog_prop["m_particle"].value / h)
        self.units["mass"] = au.solMass
        logger.info("Catalog: calculated cluster masses")
        # Cluster distances from the observer, unit: Mpc
        dist = ((self.catalog["x"]**2 + self.catalog["y"]**2 +
                 self.catalog["z"]**2) ** 0.5 *
                self.catalog_prop["l_side"].value / h)
        self.catalog["distance"] = dist
        self.units["distance"] = au.Mpc
        logger.info("Catalog: calculated cluster distances")
        # Drop unnecessary columns to save memory
        columns_drop = ["m", "sigma", "ip", "x", "y", "z", "vx", "vy", "vz"]
        self.catalog.drop(columns_drop, axis=1, inplace=True)
        logger.info("Catalog: dropped unnecessary columns: {0}".format(
            ", ".join(columns_drop)))

    def _expand_catalog_fullsky(self):
        """Expand the catalog to be full sky, by assuming that clusters are
        uniformly distributed.  Also, the radio halo fraction is also
        considered to determine the final number of clusters on the full sky.
        """
        fullsky = 4*np.pi * au.sr
        factor = float(fullsky / self.catalog_prop["coverage"])
        n0_cluster = len(self.catalog)
        logger.info("Radio halo fraction in clusters: {0}".format(
            self.halo_fraction))
        # Total number of clusters on the full sky
        N_cluster = int(n0_cluster * factor * self.halo_fraction)
        logger.info("Total number of clusters on the full sky: {0:,}".format(
            N_cluster))
        logger.info("Expanding the catalog to be full sky ...")
        idx = np.round(np.random.uniform(low=0, high=n0_cluster-1,
                                         size=N_cluster)).astype(np.int)
        self.catalog = self.catalog.iloc[idx, :]
        self.catalog.reset_index(inplace=True)
        logger.info("Done expand the catalog to be full sky")

    def _add_random_position(self):
        """Add random positions for each cluster as columns "glon" and
        "glat" to the catalog data.

        Column "glon" is the Galactic longitudes, [0, 360) (degree).
        Column "glat" is the Galactic latitudes, [-90, 90] (degree).

        The positions are uniformly distributed on the spherical surface.
        """
        logger.info("Randomly generating positions for each cluster ...")
        num = len(self.catalog)
        theta, phi = spherical_uniform(num)
        glon = np.degrees(phi)
        glat = 90.0 - np.degrees(theta)
        self.catalog["glon"] = glon
        self.catalog["glat"] = glat
        logger.info("Done add random positions for each cluster")

    def _add_random_eccentricity(self):
        """Add random eccentricities for each cluster as column
        "eccentricity" to the catalog data.

        The eccentricity of a ellipse is defined as:
            e = sqrt((a^2 - b^2) / a^2) = f / a
        where f is the distance from the center to either focus:
            f = sqrt(a^2 - b^2)

        NOTE
        ----
        The eccentricities are randomly generated from a *squared*
        standard normalization distribution, and with an upper limit
        at 0.9, i.e., the eccentricities are [0, 0.9].
        """
        logger.info("Adding random eccentricities for each cluster ...")
        num = len(self.catalog)
        eccentricity = np.random.normal(size=num) ** 2
        # Replace values beyond the upper limit by sampling from valid values
        ulimit = 0.9
        idx_invalid = (eccentricity > ulimit)
        num_invalid = idx_invalid.sum()
        eccentricity[idx_invalid] = np.random.choice(
            eccentricity[~idx_invalid], size=num_invalid)
        self.catalog["eccentricity"] = eccentricity
        logger.info("Done add random eccentricities to catalog")

    def _add_random_rotation(self):
        """Add random rotation angles for each cluster as column "rotation"
        to the catalog data.

        The rotation angles are uniformly distributed within [0, 360).

        The rotation happens on the spherical surface, i.e., not with respect
        to the line of sight, but to the Galactic frame coordinate axes.
        """
        logger.info("Adding random rotation angles for each cluster ...")
        num = len(self.catalog)
        rotation = np.random.uniform(low=0.0, high=360.0, size=num)
        self.catalog["rotation"] = rotation
        self.units["rotation"] = au.deg
        logger.info("Done add random rotation angles to catalog")

    def _calc_sizes(self):
        """Calculate the virial radii for each cluster from the masses,
        and then calculate the elliptical angular sizes by considering
        the added random eccentricities.

        Attributes
        ----------
        catalog["r_vir"] : 1D `~numpy.ndarray`
            The virial radii (unit: Mpc) calculated from the cluster masses
        catalog["size_major"], catalog["size_minor] : 1D `~numpy.ndarray`
            The major and minor axes (unit: degree) of the clusters calculated
            from the above virial radii and the random eccentricities.

        NOTE
        ----
        The elliptical major and minor axes are calculated by assuming
        the equal area between the ellipse and corresponding circle.
            theta2 = r_vir / distance
            pi * a * b = pi * (theta2)^2
            e = sqrt((a^2 - b^2) / a^2)
        thus,
            a = theta2 / (1-e^2)^(1/4)
            b = theta2 * (1-e^2)^(1/4)
        """
        logger.info("Calculating the virial radii ...")
        overdensity = self.catalog_prop["overdensity"]
        rho_crit = self.cosmo.critical_density(self.catalog["redshift"])
        mass = self.catalog["mass"].data * self.units["mass"]
        r_vir = (3 * mass / (4*np.pi * overdensity * rho_crit)) ** (1.0/3.0)
        self.catalog["r_vir"] = r_vir.to(au.Mpc).value
        self.units["r_vir"] = au.Mpc
        logger.info("Done calculate the virial radii")
        # Calculate (elliptical) angular sizes, i.e., major and minor axes
        logger.info("Calculating the elliptical angular sizes ...")
        distance = self.catalog["distance"].data * self.units["distance"]
        theta2 = (r_vir / distance).decompose().value  # [ rad ]
        size_major = theta2 / (1 - self.catalog["eccentricity"]**2) ** 0.25
        size_minor = theta2 * (1 - self.catalog["eccentricity"]**2) ** 0.25
        self.catalog["size_major"] = size_major * au.rad.to(au.deg)
        self.catalog["size_minor"] = size_minor * au.rad.to(au.deg)
        self.units["size"] = au.deg
        logger.info("Done calculate the elliptical angular sizes")

    def _calc_luminosity(self):
        """Calculate the radio luminosity (at 1.4 GHz) using empirical
        scaling relations.

        First, calculate the X-ray luminosity L_X using the empirical
        scaling relation between mass and X-ray luminosity.
        Then, derive the radio luminosity by employing the scaling
        relation between X-ray and radio luminosity.

        Attributes
        ----------
        catalog["luminosity"] : 1D `~numpy.ndarray`
            The luminosity density (at 1.4 GHz) of each cluster.
        catalog_prop["luminosity_freq"] : `~astropy.units.Quantity`
            The frequency (as an ``astropy`` quantity) where the above
            luminosity derived.
        units["luminosity"] : `~astropy.units.Unit`
            The unit used by the above luminosity.

        XXX/TODO
        --------
        The scaling relations used here may be outdated, and some of the
        quantities need trick conversions, which cause much confusion.

        Investigate for *more up-to-date scaling relations*, derived with
        new observation constraints.

        NOTE
        ----
        - The mass in the mass-X-ray luminosity scaling relation is NOT the
          cluster real mass, since [Reiprich2002]_ refer to the
          *critical density* ρ_c, while the scaling relation from
          Jenkins et al. (2001) requires mass refer to the
          *cosmic mean mass density ρ_m = Ω_m * ρ_c,
          therefore, the mass needs following conversion (which is an
          approximation):
              M_{R&B} ≈ M * sqrt(OmegaM0)
        - The derived X-ray luminosity is for the 0.1-2.4 keV energy band.
        - The X-ray-radio luminosity scaling relation adopted here is
          derived at 1.4 GHz.
        - [EnBlin2002]_ assumes H0 = 50 h50 km/s/Mpc, so h50 = 1.

        References
        ----------
        - [Jelic2008], Eq.(13,14)
        - [EnBlin2002], Eq.(1,3)
        """
        # Dimensionless Hubble parameter adopted here and in the literature
        h_our = self.H0 / 100
        h_others = 50.0 / 100
        #
        logger.info("Calculating the radio luminosity (at 1.4 GHz) ...")
        # Calculate the X-ray luminosity from mass
        # NOTE: mass conversion (see also the above notes)
        mass_RB = (self.catalog["mass"].data * self.units["mass"] *
                   self.catalog_prop["omega_m"]**0.5)
        a_X = 0.449
        b_X = 1.9
        # Hubble parameter conversion factor
        h_conv1 = (h_our / h_others) ** (b_X-2)
        # X-ray luminosity (0.1-2.4 keV) [ erg/s ]
        L_X = ((a_X * 1e45 *
                (mass_RB / (1e15*au.solMass)).decompose().value ** b_X) *
               h_conv1)
        # Calculate the radio luminosity from X-ray luminosity
        a_r = 2.78
        b_r = 1.94
        # Hubble parameter conversion factor
        h_conv2 = (h_our / h_others) ** (2*b_r-2)
        # Radio luminosity density (at 1.4 GHz) [ W/Hz ]
        L_r = (a_r * 1e24 * (L_X / 1e45)**b_r) * h_conv2
        self.catalog["luminosity"] = L_r
        self.catalog_prop["luminosity_freq"] = 1400 * self.freq_unit
        self.units["luminosity"] = au.W / au.Hz
        logger.info("Done Calculate the radio luminosity")

    def _calc_specindex(self):
        """Calculate the radio spectral indexes for each cluster.

        Attributes
        ----------
        catalog["specindex"] : 1D `~numpy.ndarray`
            The radio spectral index of each cluster.

        XXX/TODO
        --------
        Currently, a common/uniform spectral index (1.2) is assumed for all
        clusters, which may be improved by investigating more recent results.
        """
        specindex = 1.2
        logger.info("Use common spectral index for all clusters: "
                    "{0}".format(specindex))
        self.catalog["specindex"] = specindex

    def _calc_Tb(self, luminosity, distance, specindex, frequency, size):
        """Calculate the brightness temperature at requested frequency
        by assuming a power-law spectral shape.

        Parameters
        ----------
        luminosity : float
            The luminosity density (unit: [ W/Hz ]) at the reference
            frequency (i.e., `self.catalog_prop["luminosity_freq"]`).
        distance : float
            The luminosity distance (unit: [ Mpc ]) to the object
        specindex : float
            The spectral index of the power-law spectrum.
            Note the *negative* sign in the formula.
        frequency : float
            The frequency (unit: [ MHz ]) where the brightness
            temperature requested.
        size : 2-float tuple
            The (major, minor) axes (unit: [ deg ]).
            The order of major and minor can be arbitrary.

        Returns
        -------
        Tb : float
            Brightness temperature at the requested frequency, unit [ K ]

        NOTE
        ----
        The power-law spectral shape is assumed for *flux density* other
        than the *brightness temperature*.
        Therefore, the flux density at the requested frequency should first
        be calculated by extrapolating the spectrum, then convert the flux
        density to derive the brightness temperature.

        XXX/NOTE
        --------
        The *luminosity distance* is required to calculate the flux density
        from the luminosity density.
        Whether the distance (i.e., ``self.catalog["distance"]``) is the
        *comoving distance* ??
        Whether a conversion is required to get the *luminosity distance* ??
        """
        freq = frequency  # [ MHz ]
        freq_ref = self.catalog_prop["luminosity_freq"].value
        Lnu = luminosity * (freq / freq_ref) ** (-specindex)  # [ W/Hz ]
        # Conversion coefficient: [ W/Hz/Mpc^2 ] => [ Jy ]
        coef = 1.0502650403056097e-19
        Fnu = coef * Lnu / (4*np.pi * distance**2)  # [ Jy ]
        omega = size[0] * size[1]  # [ deg^2 ]
        Tb = Fnu_to_Tb_fast(Fnu, omega, freq)
        return Tb

    def _simulate_templates(self):
        """Simulate the template (HEALPix) images for each cluster, and
        cache these templates within the class.

        The template images/maps have values of (or approximate) ones for
        these effective pixels, excluding the pixels corresponding to the
        edges of original rotated ellipse, which may have values of
        significantly less than 1 due to the rotation.

        Therefore, simulating the HEALPix map of one cluster at a requested
        frequency is simply multiplying the cached template image by the
        calculated brightness temperature (Tb) at that frequency.

        Furthermore, the total HEALPix map of all clusters are straightforward
        additions of all the maps of each cluster.

        Attributes
        ----------
        templates : list
            A List containing the simulated templates for each cluster.
            Each element is a `(hpidx, hpval)` tuple with `hpidx` the
            indexes of effective HEALPix pixels (RING ordering) and `hpval`
            the values of the corresponding pixels.
            e.g., ``[ (hpidx1, hpval1), (hpidx2, hpval2), ... ]``
        """
        logger.info("Simulating HEALPix templates for each cluster ...")
        templates = []
        resolution = self.resolution.to(au.deg).value
        # Make sure the index is reset, therefore, the *row indexes* can be
        # simply used to identify the corresponding template image.
        self.catalog.reset_index(inplace=True)
        # XXX/TODO: be parallel
        for row in self.catalog.itertuples():
            # TODO: progress bar
            center = (row.glon, row.glat)
            size = (row.size_major, row.size_minor)  # already [ degree ]
            rotation = row.rotation  # already [ degree ]
            grid = make_grid_ellipse(center, size, resolution, rotation)
            hpidx, hpval = map_grid_to_healpix(grid, self.nside)
            templates.append((hpidx, hpval))
        logger.info("Done simulate %d cluster templates" % len(templates))
        self.templates = templates

    def _simulate_single(self, data, frequency):
        """Simulate one single cluster at the specified frequency, based
        on the cached template image.

        Parameters
        ----------
        data : namedtuple
            The data of the SNR to be simulated, given in a ``namedtuple``
            object, from which can get the required properties by
            ``data.key``.
            e.g., elements of `self.catalog.itertuples()`
        frequency : float
            The simulation frequency (unit: `self.freq_unit`).

        Returns
        -------
        hpidx : 1D `~numpy.ndarray`
            The indexes (in RING ordering) of the effective HEALPix
            pixels for the SNR.
        hpval : 1D `~numpy.ndarray`
            The values (i.e., brightness temperature) of each HEALPix
            pixel with respect the above indexes.

        See Also
        --------
        `self._simulate_template()` for more detailed description.
        """
        index = data.Index
        hpidx, hpval = self.templates[index]
        # Calculate the brightness temperature
        luminosity = data.luminosity
        distance = data.distance
        specindex = data.specindex
        size = (data.size_major, data.size_minor)
        Tb = self._calc_Tb(luminosity, distance, specindex, frequency, size)
        hpval = hpval * Tb
        return (hpidx, hpval)

    def _make_filepath(self, **kwargs):
        """Make the path of output file according to the filename pattern
        and output directory loaded from configurations.
        """
        data = {
            "prefix": self.prefix,
        }
        data.update(kwargs)
        filename = self.filename_pattern.format(**data)
        filepath = os.path.join(self.output_dir, filename)
        return filepath

    def _make_header(self):
        """Make the header with detail information (e.g., parameters and
        history) for the simulated products.
        """
        header = fits.Header()
        header["COMP"] = ("Extragalactic clusters of galaxies",
                          "Emission component")
        header["UNIT"] = ("Kelvin", "Map unit")
        header["CREATOR"] = (__name__, "File creator")
        # TODO:
        history = []
        comments = []
        for hist in history:
            header.add_history(hist)
        for cmt in comments:
            header.add_comment(cmt)
        self.header = header
        logger.info("Created FITS header")

    def output(self, hpmap, frequency):
        """Write the simulated free-free map to disk with proper header
        keywords and history.

        Returns
        -------
        filepath : str
            The (absolute) path to the output HEALPix map file.
        """
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            logger.info("Created output dir: {0}".format(self.output_dir))
        #
        filepath = self._make_filepath(frequency=frequency)
        if not hasattr(self, "header"):
            self._make_header()
        header = self.header.copy()
        header["FREQ"] = (frequency, "Frequency [ MHz ]")
        header["DATE"] = (
            datetime.now(timezone.utc).astimezone().isoformat(),
            "File creation date"
        )
        if self.use_float:
            hpmap = hpmap.astype(np.float32)
        write_fits_healpix(filepath, hpmap, header=header,
                           clobber=self.clobber, checksum=self.checksum)
        logger.info("Write simulated map to file: {0}".format(filepath))
        return filepath

    def preprocess(self):
        """Perform the preparation procedures for the final simulations.

        Attributes
        ----------
        _preprocessed : bool
            This attribute presents and is ``True`` after the preparation
            procedures are performed, which indicates that it is ready to
            do the final simulations.
        """
        if hasattr(self, "_preprocessed") and self._preprocessed:
            return
        #
        logger.info("{name}: preprocessing ...".format(name=self.name))
        self._load_catalog()
        self._process_catalog()
        #
        self._expand_catalog_fullsky()
        self._add_random_position()
        self._add_random_eccentricity()
        self._add_random_rotation()
        #
        self._calc_sizes()
        self._calc_luminosity()
        self._calc_specindex()
        #
        self._simulate_templates()
        #
        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """Simulate the emission (HEALPix) map of all Galactic SNRs at
        the specified frequency.

        Parameters
        ----------
        frequency : float
            The simulation frequency (unit: `self.freq_unit`).

        Returns
        -------
        hpmap_f : 1D `~numpy.ndarray`
            The HEALPix map (RING ordering) at the input frequency.
        filepath : str
            The (absolute) path to the output HEALPix file if saved,
            otherwise ``None``.

        See Also
        --------
        `self._simulate_template()` for more detailed description.
        """
        self.preprocess()
        #
        logger.info("Simulating {name} map at {freq} ({unit}) ...".format(
            name=self.name, freq=frequency, unit=self.freq_unit))
        hpmap_f = np.zeros(hp.nside2npix(self.nside))
        # XXX/TODO: be parallel
        for row in self.catalog.itertuples():
            # TODO: progress bar
            hpidx, hpval = self._simulate_single(row, frequency)
            hpmap_f[hpidx] += hpval
        #
        if self.save:
            filepath = self.output(hpmap_f, frequency)
        else:
            filepath = None
        return (hpmap_f, filepath)

    def simulate(self, frequencies):
        """Simulate the emission (HEALPix) maps of all Galactic SNRs for
        every specified frequency.

        Parameters
        ----------
        frequency : list[float]
            List of frequencies (unit: `self.freq_unit`) where the
            simulation performed.

        Returns
        -------
        hpmaps : list[1D `~numpy.ndarray`]
            List of HEALPix maps (in RING ordering) at each frequency.
        paths : list[str]
            List of (absolute) path to the output HEALPix maps.
        """
        hpmaps = []
        paths = []
        for f in np.array(frequencies, ndmin=1):
            hpmap_f, filepath = self.simulate_frequency(f)
            hpmaps.append(hpmap_f)
            paths.append(filepath)
        return (hpmaps, paths)

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the catalog actually used in the simulation
        self._save_catalog_inuse()
