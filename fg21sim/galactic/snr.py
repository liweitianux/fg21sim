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


logger = logging.getLogger(__name__)


class SuperNovaRemnants:
    """
    Simulate the Galactic supernova remnants emission.

    ???

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
        # TODO/XXX
        pass

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
        n_delete = cond_keep.sum()
        n_delete_p = n_delete / n_total * 100
        n_remain = n_total - n_delete
        self.catalog = self.catalog[cond_keep]
        # Reset index
        self.catalog.reset_index(inplace=True)
        logger.info("SNRs catalog: filtered out " +
                    "{0:d} ({1:.1f}) objects".format(n_delete, n_delete_p))
        logger.info("SNRs catalog: remaining {0} objects".format(n_remain))

    def _add_random_rotation(self):
        """Add random rotation angles for each SNR as column "rotation"
        within the `catalog` data frame.

        The rotation angles are uniformly distributed within [0, 360).

        The rotation happens on the spherical surface, i.e., not with respect
        to the line of sight, but to the Galactic frame coordinate axes.
        """
        num = len(self.catalog)
        angles = np.random.uniform(low=0.0, high=360.0, size=num)
        rotation = pd.Series(data=angles, name="rotation")
        self.catalog["rotation"] = rotation
        logger.info("Added random rotation angles as the 'rotation' column")

    def _calc_Tb(self, flux, specindex, frequency):
        """Calculate the brightness temperature at requested frequency
        by assuming a power-law spectral shape.

        Parameters
        ----------
        flux : float
            The flux density (unit: [ Jy ]) at the reference frequency
            (`self.catalog_flux_freq`).
        specindex : float
            The spectral index of the power-law spectrum
        frequency : float
            The frequency (unit: [ MHz ]) where the brightness temperature
            requested.

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
        pass

    def _simulate_frequency(self, frequency):
        """Simulate the Galactic SNRs emission map at the specified frequency.
        """
        pass

    def _make_filename(self, **kwargs):
        """Make the path of output file according to the filename pattern
        and output directory loaded from configurations.
        """
        data = {
            "prefix": self.prefix,
        }
        data.extend(kwargs)
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

    def simulate(self, frequencies):
        """Simulate the free-free map at the specified frequencies."""
        hpmaps = []
        for f in np.array(frequencies, ndmin=1):
            logger.info("Simulating free-free map at {0} ({1}) ...".format(
                f, self.freq_unit))
            hpmap_f = self._simulate_frequency(f)
            hpmaps.append(hpmap_f)
            if self.save:
                self.output(hpmap_f, f)
        return hpmaps
