# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Galactic supernova remnants (SNRs) emission simulations.
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
import astropy.units as au
import healpy as hp
import pandas as pd

from ..utils import write_fits_healpix
from ..utils.convert import Fnu_to_Tb
from ..utils.grid import make_grid_ellipse, map_grid_to_healpix


logger = logging.getLogger(__name__)


class SuperNovaRemnants:
    """
    Simulate the Galactic supernova remnants emission.

    The simulation is based on the Galactic SNRs catalog maintained by
    *D. A. Green* [GreenSNRDataWeb]_, which contains 294 SNRs (2014-May).
    However, some SNRs have incomplete data which are excluded, while
    some SNRs with uncertain properties are currently kept.

    Every SNR is simulated as an *ellipse* of *uniform brightness* on
    a local coordinate grid with relatively higher resolution compared
    to the output HEALPix map, which is then mapped to the output HEALPix
    map by down-sampling.

    Parameters
    ----------
    configs : ConfigManager object
        An `ConfigManager` object contains default and user configurations.
        For more details, see the example config specification.

    Attributes
    ----------
    ???

    References
    ----------
    .. [Green2014]
       Green, D. A.,
       "A catalogue of 294 Galactic supernova remnants",
       2014, Bulletin of the Astronomical Society of India, 42, 47-58,
       http://adsabs.harvard.edu/abs/2014BASI...42...47G

    .. [GreenSNRDataWeb]
       A Catalogue of Galactic Supernova Remnants
       http://www.mrao.cam.ac.uk/surveys/snrs/
    """
    # Component name
    name = "Galactic supernova remnants"

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        comp = "galactic/snr"
        self.catalog_path = self.configs.get_path(comp+"/catalog")
        self.catalog_outfile = self.configs.get_path(comp+"/catalog_outfile")
        self.resolution = self.configs.getn(comp+"/resolution") * au.arcmin
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.clobber = self.configs.getn("output/clobber")
        self.nside = self.configs.getn("common/nside")
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        #
        logger.info("Loaded and set up configurations")

    def _load_catalog(self):
        """Load the Galactic SNRs catalog data."""
        self.catalog = pd.read_csv(self.catalog_path)
        nrow, ncol = self.catalog.shape
        logger.info("Loaded SNRs catalog data from: {0}".format(
            self.catalog_path))
        logger.info("SNRs catalog data: {0} objects, {1} columns".format(
            nrow, ncol))
        # Set the units for columns
        self.units = {
            "glon": au.deg,
            "glat": au.deg,
            "size": au.arcmin,
            "flux": au.Jy,
        }
        # The flux densities are given at 1 GHz
        self.catalog_flux_freq = 1.0 * au.GHz

    def _save_catalog_inuse(self):
        """Save the effective/inuse SNRs catalog data to a CSV file.

        NOTE
        ----
        - Only the effective/inuse SNRs are saved (i.e., without the ones
          that are filtered out).
        - Also save the simulated rotation column.
        - The unnecessary columns are striped.
        """
        if self.catalog_outfile is None:
            logger.warning("Catalog output file not set, so do NOT save.")
            return
        # Create directory if necessary
        dirname = os.path.dirname(self.catalog_outfile)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            logger.info("Created directory: {0}".format(dirname))
        # Save catalog data
        colnames = ["name", "glon", "glat", "ra", "dec",
                    "size_major", "size_minor", "flux",
                    "specindex", "rotation"]
        if os.path.exists(self.catalog_outfile):
            if self.clobber:
                os.remove(self.catalog_outfile)
            else:
                raise OSError("Output file already exists: {0}".format(
                    self.catalog_outfile))
        self.catalog.to_csv(self.catalog_outfile, columns=colnames,
                            header=True, index=False)
        logger.info("Save SNRs catalog in use to: %s" % self.catalog_outfile)

    def _filter_catalog(self):
        """Filter the catalog data to remove the objects with incomplete
        data.

        The following cases are filtered out:
        - Missing angular size
        - Missing flux density data
        - Missing spectral index value

        NOTE
        ----
        The objects with uncertain data are currently kept.
        """
        cond1 = pd.isnull(self.catalog["size_major"])
        cond2 = pd.isnull(self.catalog["size_minor"])
        cond3 = pd.isnull(self.catalog["flux"])
        cond4 = pd.isnull(self.catalog["specindex"])
        cond_keep = ~(cond1 | cond2 | cond3 | cond4)
        n_total = len(cond_keep)
        n_remain = cond_keep.sum()
        n_delete = n_total - n_remain
        n_delete_p = n_delete / n_total * 100
        self.catalog = self.catalog[cond_keep]
        # Drop the index
        self.catalog.reset_index(drop=True, inplace=True)
        self.catalog_filtered = True
        logger.info("SNRs catalog: filtered out " +
                    "{0:d} ({1:.1f}%) objects".format(n_delete, n_delete_p))
        logger.info("SNRs catalog: remaining {0} objects".format(n_remain))

    def _add_random_rotation(self):
        """Add random rotation angles for each SNR as column "rotation"
        within the catalog data frame.

        The rotation angles are uniformly distributed within [0, 360).

        The rotation happens on the spherical surface, i.e., not with respect
        to the line of sight, but to the Galactic frame coordinate axes.
        """
        num = len(self.catalog)
        angles = np.random.uniform(low=0.0, high=360.0, size=num)
        rotation = pd.Series(data=angles, name="rotation")
        self.catalog["rotation"] = rotation
        self.units["rotation"] = au.deg
        logger.info("Added random rotation angles as the 'rotation' column")

    def _calc_Tb(self, flux, specindex, frequency, size):
        """Calculate the brightness temperature at requested frequency
        by assuming a power-law spectral shape.

        Parameters
        ----------
        flux : float
            The flux density (unit: `self.units["flux"]`) at the reference
            frequency (i.e., `self.catalog_flux_freq`).
        specindex : float
            The spectral index of the power-law spectrum
        frequency : float
            The frequency (unit: `self.freq_unit`) where the brightness
            temperature requested.
        size : 2-float tuple
            The (major, minor) axes of the SNR (unit: `self.units["size"]`).
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
        """
        freq = frequency * self.freq_unit
        flux = flux * self.units["flux"]
        Fnu = flux * (freq / self.catalog_flux_freq).value ** (-specindex)
        omega = size[0]*self.units["size"] * size[1]*self.units["size"]
        Tb = Fnu_to_Tb(Fnu, omega, freq)
        return Tb.value

    def _simulate_templates(self):
        """Simulate the template (HEALPix) images for each SNR, and cache
        these templates within the class.

        The template images/maps have values of (or approximate) ones for
        these effective pixels, excluding the pixels corresponding to the
        edges of original rotated ellipse, which may have values of
        significantly less than 1 due to the rotation.

        Therefore, simulating the HEALPix map of one SNR at a requested
        frequency is simply multiplying the cached template image by the
        calculated brightness temperature (Tb) at that frequency.

        Furthermore, the total HEALPix map of all SNRs are straightforward
        additions of all the maps of each SNR.

        Attributes
        ----------
        templates : dict
            A dictionary containing the simulated templates for each SNR.
            The dictionary keys are the names (`self.catalog["name"]`)
            of the SNRs, and the values are `(hpidx, hpval)` tuples with
            `hpidx` the indexes of effective HEALPix pixels (RING ordering)
            and `hpval` the values of the corresponding pixels.
            e.g.,
            ``{ name1: (hpidx1, hpval1), name2: (hpidx2, hpval2), ... }``
        """
        templates = {}
        resolution = self.resolution.to(au.deg).value
        for row in self.catalog.itertuples():
            name = row.name
            logger.info("Simulate HEALPix template for SNR: {0}".format(name))
            center = (row.glon, row.glat)
            size = ((row.size_major * self.units["size"]).to(au.deg).value,
                    (row.size_minor * self.units["size"]).to(au.deg).value)
            rotation = row.rotation
            grid = make_grid_ellipse(center, size, resolution, rotation)
            hpidx, hpval = map_grid_to_healpix(grid, self.nside)
            templates[name] = (hpidx, hpval)
        logger.info("Done simulate {0} SNR templates".format(len(templates)))
        self.templates = templates

    def _simulate_single(self, data, frequency):
        """Simulate one single SNR at the specified frequency.

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
        name = data.name
        hpidx, hpval = self.templates[name]
        # Calculate the brightness temperature
        flux = data.flux
        specindex = data.specindex
        size = (data.size_major, data.size_minor)
        Tb = self._calc_Tb(flux, specindex, frequency, size)
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
        filetype = self.configs.getn("output/filetype")
        if filetype == "fits":
            filename += ".fits"
        else:
            raise NotImplementedError("unsupported filetype: %s" % filetype)
        filepath = os.path.join(self.output_dir, filename)
        return filepath

    def _make_header(self):
        """Make the header with detail information (e.g., parameters and
        history) for the simulated products.
        """
        header = fits.Header()
        header["COMP"] = ("Galactic supernova remnants (SNRs)",
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
                           clobber=self.clobber)
        logger.info("Write simulated map to file: {0}".format(filepath))

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
        self._filter_catalog()
        self._add_random_rotation()
        # Simulate the template maps for each SNR
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
            HEALPix map data in RING ordering

        See Also
        --------
        `self._simulate_template()` for more detailed description.
        """
        self.preprocess()
        #
        logger.info("Simulating {name} map at {freq} ({unit}) ...".format(
            name=self.name, freq=frequency, unit=self.freq_unit))
        hpmap_f = np.zeros(hp.nside2npix(self.nside))
        for row in self.catalog.itertuples():
            hpidx, hpval = self._simulate_single(row, frequency)
            hpmap_f[hpidx] += hpval
        #
        if self.save:
            self.output(hpmap_f, frequency)
        return hpmap_f

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
        """
        hpmaps = []
        for f in np.array(frequencies, ndmin=1):
            hpmap_f = self.simulate_frequency(f)
            hpmaps.append(hpmap_f)
        return hpmaps

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("{name}: postprocessing ...".format(name=self.name))
        # Save the catalog actually used in the simulation
        self._save_catalog_inuse()
