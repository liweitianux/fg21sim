# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Diffuse Galactic synchrotron emission (unpolarized) simulations.
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
import astropy.units as au
import healpy as hp

from ..utils import read_fits_healpix, write_fits_healpix


logger = logging.getLogger(__name__)


class Synchrotron:
    """
    Simulate the diffuse Galactic synchrotron emission based on an
    existing template.

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
    ???
    """
    # Component name
    name = "Galactic free-free"

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        comp = "galactic/synchrotron"
        self.template_path = self.configs.get_path(comp+"/template")
        self.template_freq = self.configs.getn(comp+"/template_freq")
        self.template_unit = au.Unit(
            self.configs.getn(comp+"/template_unit"))
        self.indexmap_path = self.configs.get_path(comp+"/indexmap")
        self.smallscales = self.configs.getn(comp+"/add_smallscales")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        # output
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        self.nside = self.configs.getn("common/nside")
        self.lmin = self.configs.getn("common/lmin")
        self.lmax = self.configs.getn("common/lmax")
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        #
        logger.info("Loaded and setup configurations")

    def _load_template(self):
        """Load the template map, and upgrade/downgrade the resolution
        to match the output Nside.
        """
        self.template, self.template_header = read_fits_healpix(
            self.template_path)
        template_nside = self.template_header["NSIDE"]
        logger.info("Loaded template map from {0} (Nside={1})".format(
            self.template_path, template_nside))
        # Upgrade/downgrade resolution
        if template_nside != self.nside:
            self.template = hp.ud_grade(self.template, nside_out=self.nside)
            logger.info("Upgrade/downgrade template map from Nside "
                        "{0} to {1}".format(template_nside, self.nside))

    def _load_indexmap(self):
        """Load the spectral index map, and upgrade/downgrade the resolution
        to match the output Nside.
        """
        self.indexmap, self.indexmap_header = read_fits_healpix(
            self.indexmap_path)
        indexmap_nside = self.indexmap_header["NSIDE"]
        logger.info("Loaded spectral index map from {0} (Nside={1})".format(
            self.indexmap_path, indexmap_nside))
        # Upgrade/downgrade resolution
        if indexmap_nside != self.nside:
            self.indexmap = hp.ud_grade(self.indexmap, nside_out=self.nside)
            logger.info("Upgrade/downgrade spectral index map from Nside "
                        "{0} to {1}".format(indexmap_nside, self.nside))

    def _add_smallscales(self):
        """Add fluctuations on small scales to the template map.

        XXX/TODO:
        * Support using different models.
        * This should be extensible/plug-able, e.g., a separate module
          and allow easily add new models for use.

        References
        ----------
        [1] M. Remazeilles et al. 2015, MNRAS, 451, 4311-4327
            "An improved source-subtracted and destriped 408-MHz all-sky map"
            Sec. 4.2: Small-scale fluctuations
        """
        if (not self.smallscales) or (hasattr(self, "hpmap_smallscales")):
            return
        # To add small scale fluctuations
        # model: Remazeilles15
        gamma = -2.703  # index of the power spectrum between l [30, 90]
        sigma_tp = 56  # original beam resolution of the template [ arcmin ]
        alpha = 0.0599
        beta = 0.782
        # angular power spectrum of the Gaussian random field
        ell = np.arange(self.lmax+1).astype(np.int)
        cl = np.zeros(ell.shape)
        ell_idx = ell >= self.lmin
        cl[ell_idx] = (ell[ell_idx] ** gamma *
                       1.0 - np.exp(-ell[ell_idx]**2 * sigma_tp**2))
        cl[ell < self.lmin] = cl[self.lmin]
        # generate a realization of the Gaussian random field
        gss = hp.synfast(cls=cl, nside=self.nside, new=True)
        # whiten the Gaussian random field
        gss = (gss - gss.mean()) / gss.std()
        self.hpmap_smallscales = alpha * gss * self.template**beta
        self.template += self.hpmap_smallscales
        logger.info("Added small-scale fluctuations")

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
        header["COMP"] = ("Galactic synchrotron (unpolarized)",
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
        """Write the simulated synchrotron map to disk with proper
        header keywords and history.
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
        self._load_template()
        self._load_indexmap()
        self._add_smallscales()
        #
        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """Transform the template map to the requested frequency,
        according to the spectral model and using an spectral index map.
        """
        self.preprocess()
        #
        logger.info("Simulating {name} map at {freq} ({unit}) ...".format(
            name=self.name, freq=frequency, unit=self.freq_unit))
        hpmap_f = (self.template *
                   (frequency / self.template_freq) ** self.indexmap)
        #
        if self.save:
            self.output(hpmap_f, frequency)
        return hpmap_f

    def simulate(self, frequencies):
        """Simulate the synchrotron map at the specified frequencies."""
        hpmaps = []
        for f in np.array(frequencies, ndmin=1):
            hpmap_f = self.simulate_frequency(f)
            hpmaps.append(hpmap_f)
        return hpmaps

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        pass
