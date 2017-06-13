# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
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

from ..sky import get_sky


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
    name = "Galactic synchrotron (unpolarized)"

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
        self.add_smallscales = self.configs.getn(comp+"/add_smallscales")
        self.smallscales_added = False
        self.lmin = self.configs.getn(comp+"/lmin")
        self.lmax = self.configs.getn(comp+"/lmax")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        # output
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        #
        logger.info("Loaded and setup configurations")

    def _load_maps(self):
        """Load the template map and spectral index map."""
        sky = get_sky(self.configs)
        logger.info("Loading template map ...")
        self.template = sky.load(self.template_path)
        logger.info("Loading spectral index map ...")
        self.indexmap = sky.load(self.indexmap_path)

    def _add_smallscales(self):
        """
        Add fluctuations on small scales to the template map.

        NOTE:
        Only when the input template is a HEALPix map, this function
        will be applied to add the small-scale fluctuations, which assuming
        a angular power spectrum model.

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
        if (not self.add_smallscales) or (self.smallscales_added):
            return
        if self.template.type_ != "healpix":
            logger.warning("Input template map is NOT a HEALPix map; " +
                           "skip adding small-scale fluctuations!")
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
        gss = hp.synfast(cls=cl, nside=self.template.nside, new=True)
        # whiten the Gaussian random field
        gss = (gss - gss.mean()) / gss.std()
        hpmap_smallscales = alpha * gss * self.template.data**beta
        self.template.data += hpmap_smallscales
        logger.info("Added small-scale fluctuations to template map")

    def _make_filepath(self, **kwargs):
        """
        Make the path of output file according to the filename pattern
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
        """
        Make the header with detail information (e.g., parameters and
        history) for the simulated products.
        """
        header = fits.Header()
        header["COMP"] = (self.name, "Emission component")
        header["BUNIT"] = ("K", "data unit is Kelvin")
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

    def output(self, skymap, frequency):
        """
        Write the simulated synchrotron map to disk with proper
        header keywords and history.

        Returns
        -------
        outfile : str
            The (absolute) path to the output sky map file.
        """
        outfile = self._make_filepath(frequency=frequency)
        if not hasattr(self, "header"):
            self._make_header()
        header = self.header.copy()
        header["FREQ"] = (frequency, "Frequency [ MHz ]")
        header["DATE"] = (
            datetime.now(timezone.utc).astimezone().isoformat(),
            "File creation date"
        )
        if self.use_float:
            skymap = skymap.astype(np.float32)
        sky = get_sky(configs=self.configs)
        sky.data = skymap
        sky.header = header
        sky.write(outfile, clobber=self.clobber, checksum=self.checksum)
        return outfile

    def preprocess(self):
        """
        Perform the preparation procedures for the final simulations.

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
        self._load_maps()
        self._add_smallscales()
        #
        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """
        Transform the template map to the requested frequency,
        according to the spectral model and using an spectral index map.

        Returns
        -------
        skymap_f : 1D `~numpy.ndarray`
            The sky map at the input frequency.
        filepath : str
            The (absolute) path to the output sky map if saved,
            otherwise ``None``.
        """
        self.preprocess()
        #
        logger.info("Simulating {name} map at {freq} ({unit}) ...".format(
            name=self.name, freq=frequency, unit=self.freq_unit))
        skymap_f = (self.template.data *
                    (frequency / self.template_freq) ** self.indexmap.data)
        #
        if self.save:
            filepath = self.output(skymap_f, frequency)
        else:
            filepath = None
        return (skymap_f, filepath)

    def simulate(self, frequencies):
        """
        Simulate the synchrotron map at the specified frequencies.

        Returns
        -------
        skymaps : list[1D `~numpy.ndarray`]
            List of sky maps at each frequency.
        paths : list[str]
            List of (absolute) path to the output sky maps.
        """
        skymaps = []
        paths = []
        for f in np.array(frequencies, ndmin=1):
            skymap_f, outfile = self.simulate_frequency(f)
            skymaps.append(skymap_f)
            paths.append(outfile)
        return (skymaps, paths)

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        pass
