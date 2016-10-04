# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Diffuse Galactic synchrotron emission (unpolarized) simulations.
"""

import os
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
import astropy.units as au
import healpy as hp


class Synchrotron:
    """Simulate the diffuse Galactic synchrotron emission based on an
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
    def __init__(self, configs):
        self.configs = configs
        self._set_configs()
        self._load_template()
        self._load_indexmap()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        self.template_path = self.configs.getn(
            "galactic/synchrotron/template")
        self.template_freq = self.configs.getn(
            "galactic/synchrotron/template_freq")
        self.template_unit = au.Unit(
            self.configs.getn("galactic/synchrotron/template_unit"))
        self.indexmap_path = self.configs.getn(
            "galactic/synchrotron/indexmap")
        self.smallscales = self.configs.getn(
            "galactic/synchrotron/add_smallscales")
        # output
        self.prefix = self.configs.getn("galactic/synchrotron/prefix")
        self.save = self.configs.getn("galactic/synchrotron/save")
        self.output_dir = self.configs.getn("galactic/synchrotron/output_dir")
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.clobber = self.configs.getn("output/clobber")
        # common
        self.nside = self.configs.getn("common/nside")
        self.lmin = self.configs.getn("common/lmin")
        self.lmax = self.configs.getn("common/lmax")
        # unit of the frequency
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))

    def _load_template(self):
        """Load the template map"""
        self.template, header = hp.read_map(self.template_path,
                                            nest=False, h=True, verbose=False)
        self.template_header = fits.header.Header(header)
        self.template_nside = self.template_header["NSIDE"]
        self.template_ordering = self.template_header["ORDERING"]

    def _load_indexmap(self):
        """Load the spectral index map"""
        self.indexmap, header = hp.read_map(self.indexmap_path,
                                            nest=False, h=True, verbose=False)
        self.indexmap_header = fits.header.Header(header)
        self.indexmap_nside = self.indexmap_header["NSIDE"]
        self.indexmap_ordering = self.indexmap_header["ORDERING"]

    def _process_template(self):
        """Upgrade/downgrade the template to have the same Nside as
        requested by the output."""
        if self.nside == self.template_nside:
            self.hpmap = self.template.copy()
        else:
            # upgrade/downgrade the resolution
            self.hpmap = hp.ud_grade(self.template, nside_out=self.nside)

    def _process_indexmap(self):
        """Upgrade/downgrade the spectral index map to have the same
        Nside as requested by the output."""
        if self.nside == self.indexmap_nside:
            self.hpmap_index = self.indexmap.copy()
        else:
            # upgrade/downgrade the resolution
            self.hpmap_index = hp.ud_grade(self.indexmap, nside_out=self.nside)

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
        if not self.smallscales:
            return
        # To add small scale fluctuations
        # model: Remazeilles15
        gamma = -2.703  # index of the power spectrum between l [30, 90]
        sigma_temp = 56  # original beam resolution of the template [ arcmin ]
        alpha = 0.0599
        beta = 0.782
        # angular power spectrum of the Gaussian random field
        ell = np.arange(self.lmax+1).astype(np.int)
        cl = np.zeros(ell.shape)
        ell_idx = ell >= self.lmin
        cl[ell_idx] = (ell[ell_idx] ** gamma *
                       1.0 - np.exp(-ell[ell_idx]**2 * sigma_temp**2))
        cl[ell < self.lmin] = cl[self.lmin]
        # generate a realization of the Gaussian random field
        gss = hp.synfast(cls=cl, nside=self.nside)
        # whiten the Gaussian random field
        gss = (gss - gss.mean()) / gss.std()
        self.hpmap_smallscales = alpha * gss * self.hpmap**beta
        self.hpmap += self.hpmap_smallscales

    def _transform_frequency(self, frequency):
        """Transform the template map to the requested frequency,
        according to the spectral model and using an spectral index map.
        """
        hpmap_f = (self.hpmap *
                   (frequency / self.template_freq) ** self.hpmap_index)
        return hpmap_f

    def _make_header(self):
        """Make the header with detail information (e.g., parameters and
        history) for the simulated products.
        """
        header = fits.Header()
        header["COMP"] = ("Galactic synchrotron (unpolarized)",
                          "Emission component")
        # TODO:
        history = []
        comments = []
        for hist in history:
            header.add_history(hist)
        for cmt in comments:
            header.add_comment(cmt)
        self.header = header

    def output(self, hpmap, frequency):
        """Write the simulated synchrotron map to disk with proper
        header keywords and history.
        """
        FITS_COLUMN_FORMATS = {
            np.dtype("float32"): "E",
            np.dtype("float64"): "D",
        }
        #
        filename = self.filename_pattern.format(prefix=self.prefix,
                                                frequency=frequency)
        filepath = os.path.join(self.output_dir, filename)
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
        hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="I", array=hpmap,
                        format=FITS_COLUMN_FORMATS.get(hpmap.dtype))
        ], header=header)
        hdu.writeto(filepath, clobber=self.clobber, checksum=True)

    def simulate(self, frequencies):
        """Simulate the synchrotron map at the specified frequencies."""
        if not hasattr(self, "hpmap"):
            self._process_template()
            self._add_smallscales()
        if not hasattr(self, "hpmap_index"):
            self._process_indexmap()
        #
        hpmaps = []
        for f in np.array(frequencies, ndmin=1):
            hpmap_f = self._transform_frequency(f)
            hpmaps.append(hpmap_f)
            if self.save:
                self.output(hpmap_f, f)
        return hpmaps
