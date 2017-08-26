# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Diffuse Galactic synchrotron emission (unpolarized) simulations.
"""

import os
import logging

import numpy as np
import healpy as hp

from ..share import CONFIGS
from ..sky import get_sky


logger = logging.getLogger(__name__)


class Synchrotron:
    """
    Simulate the diffuse Galactic synchrotron emission based on an
    existing template.

    Parameters
    ----------
    configs : `~ConfigManager`
        An `ConfigManager` object contains default and user configurations.
        For more details, see the example config specification.

    Attributes
    ----------
    sky : `~SkyBase`
        The sky instance to deal with the simulation sky as well as the
        output map.

    References
    ----------
    ???
    """
    # Component name
    compID = "galactic/synchrotron"
    name = "Galactic synchrotron (unpolarized)"

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
        self.template_path = self.configs.get_path(comp+"/template")
        self.template_freq = self.configs.getn(comp+"/template_freq")
        self.indexmap_path = self.configs.get_path(comp+"/indexmap")
        self.add_smallscales = self.configs.getn(comp+"/add_smallscales")
        self.smallscales_added = False
        self.lmin = self.configs.getn(comp+"/lmin")
        self.lmax = self.configs.getn(comp+"/lmax")
        self.prefix = self.configs.getn(comp+"/prefix")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        # output
        self.frequencies = self.configs.frequencies  # [MHz]
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.clobber = self.configs.getn("output/clobber")
        #
        logger.info("Loaded and setup configurations")

    def _load_maps(self):
        """Load the template map and spectral index map."""
        logger.info("Loading template map ...")
        self.template = self.sky.open(self.template_path)
        logger.info("Loading spectral index map ...")
        self.indexmap = self.sky.open(self.indexmap_path)

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

        # Parameters to extrapolate the angular power spectrum
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
        self.template += hpmap_smallscales
        logger.info("Added small-scale fluctuations to template map")

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
        Perform the preparation procedures for the final simulations.

        Attributes
        ----------
        _preprocessed : bool
            This attribute presents and is ``True`` after the preparation
            procedures are performed, which indicates that it is ready to
            do the subsequent simulations.
        """
        if hasattr(self, "_preprocessed") and self._preprocessed:
            return

        logger.info("{name}: preprocessing ...".format(name=self.name))
        self._load_maps()
        self._add_smallscales()

        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """
        Simulate the emission at requested lower frequency by
        extrapolating the template map using the power-law model and
        the spectral index map.

        Parameters
        ----------
        frequency : float
            The frequency where to simulate the radio emission.
            Unit: [MHz]

        Returns
        -------
        sky : `~SkyBase`
            The simulated sky image as a new sky instance.
        """
        logger.info("Simulating {name} map at {freq:.2f} [MHz] ...".format(
            name=self.name, freq=frequency))
        sky = self.sky.copy()
        sky.frequency = frequency
        ff = frequency / self.template_freq
        data = self.template * ff ** (-np.abs(self.indexmap))
        sky.data = data
        logger.info("Done simulate map at %.2f [MHz]." % frequency)
        return sky

    def simulate(self, frequencies=None):
        """
        Simulate the synchrotron map at the specified frequencies.

        Parameters
        ----------
        frequencies : float, or list[float]
            The frequencies where to simulate the foreground map.
            Unit: [MHz]
            Default: None (i.e., use ``self.frequencies``)
        """
        if frequencies is None:
            frequencies = self.frequencies
        else:
            frequencies = np.array(frequencies, ndmin=1)

        logger.info("Simulating {name} ...".format(name=self.name))
        for freq in frequencies:
            sky = self.simulate_frequency(freq)
            outfile = self._outfilepath(frequency=freq)
            sky.write(outfile)
        logger.info("Done simulate {name}!".format(name=self.name))

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("{name}: postprocessing ...".format(name=self.name))
        logger.info("^_^ nothing to do :-)")
