# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Diffuse Galactic free-free emission simulations.

References
----------
.. [dickinson2003]
   Dickinson, C.; Davies, R. D.; Davis, R. J.,
   "Towards a free-free template for CMB foregrounds",
   2003, MNRAS, 341, 369,
   http://adsabs.harvard.edu/abs/2003MNRAS.341..369D

.. [finkbeiner2003]
   Finkbeiner, Douglas P.,
   "A Full-Sky Hα Template for Microwave Foreground Prediction",
   2003, ApJS, 146, 407,
   http://adsabs.harvard.edu/abs/2003ApJS..146..407F

.. [schlegel1998]
   Schlegel, David J.; Finkbeiner, Douglas P.; Davis, Marc,
   "Maps of Dust Infrared Emission for Use in Estimation of Reddening
   and Cosmic Microwave Background Radiation Foregrounds",
   1998, ApJ, 500, 525,
   http://adsabs.harvard.edu/abs/1998ApJ...500..525S
"""

import os
import logging

import numpy as np
import astropy.units as au

from ..sky import get_sky


logger = logging.getLogger(__name__)


class FreeFree:
    """
    Simulate the diffuse Galactic free-free emission.

    The [dickinson2003] method is followed to derive the free-free template.
    The all-sky Hα survey map [Finkbeiner2003] is first corrected for dust
    absorption using the infrared 100-μm dust map [Schlegel1998],
    and then converted to free-free emission map (brightness temperature).

    Parameters
    ----------
    configs : `~ConfigManager`
        An ``ConfigManager`` object contains default and user configurations.
        For more details, see the example config specification.

    Attributes
    ----------
    TODO
    """
    # Component name
    comp = "galactic/freefree"
    name = "Galactic free-free"

    def __init__(self, configs):
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
        self.halphamap_path = self.configs.get_path(comp+"/halphamap")
        self.halphamap_unit = au.Unit(
            self.configs.getn(comp+"/halphamap_unit"))
        self.dustmap_path = self.configs.get_path(comp+"/dustmap")
        self.dustmap_unit = au.Unit(
            self.configs.getn(comp+"/dustmap_unit"))
        self.f_dust = self.configs.getn(comp+"/dust_fraction")
        self.halpha_abs_th = self.configs.getn(comp+"/halpha_abs_th")  # [mag]
        self.Te = self.configs.getn(comp+"/electron_temperature")  # [K]
        self.prefix = self.configs.getn(comp+"/prefix")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.clobber = self.configs.getn("output/clobber")
        self.frequencies = self.configs.frequencies  # [MHz]
        #
        logger.info("Loaded and set up configurations")

    def _load_maps(self):
        """
        Load the Hα map, and 100-μm dust map.
        """
        logger.info("Loading H[alpha] map ...")
        self.halphamap = self.sky.open(self.halphamap_path)
        # Validate input map unit
        if self.halphamap_unit != au.Unit("Rayleigh"):
            raise ValueError("unsupported Halpha map unit: {0}".format(
                self.halphamap_unit))
        logger.info("Loading dust map ...")
        self.dustmap = self.sky.open(self.dustmap_path)
        # Validate input map unit
        if self.dustmap_unit != au.Unit("MJy / sr"):
            raise ValueError("unsupported dust map unit: {0}".format(
                self.dustmap_unit))

    def _correct_dust_absorption(self):
        """
        Correct the Hα map for dust absorption using the
        100-μm dust map.

        References: Ref.[dickinson2003],Eq.(1,3),Sec.(2.5)
        """
        if hasattr(self, "_dust_corrected") and self._dust_corrected:
            return

        logger.info("Correct H[alpha] map for dust absorption")
        logger.info("Effective dust fraction: {0}".format(self.f_dust))
        # Mask the regions where the true Halpha absorption is uncertain.
        # When the dust absorption goes rather large, the true Halpha
        # absorption can not well determined.
        # Corresponding dust absorption threshold, unit: [ MJy / sr ]
        dust_abs_th = self.halpha_abs_th / 0.0462 / self.f_dust
        logger.info("Dust absorption mask threshold: " +
                    "{0:.1f} MJy/sr ".format(dust_abs_th) +
                    "<-> H[alpha] absorption threshold: " +
                    "{0:.1f} mag".format(self.halpha_abs_th))
        mask = (self.dustmap.data > dust_abs_th)
        self.dustmap.data[mask] = np.nan
        fp_mask = 100 * mask.sum() / self.dustmap.data.size
        logger.warning("Dust map masked fraction: {0:.1f}%".format(fp_mask))
        #
        halphamap_corr = (self.halphamap.data *
                          10**(self.dustmap.data * 0.0185 * self.f_dust))
        self.halphamap.data = halphamap_corr
        self._dust_corrected = True
        logger.info("Done dust absorption correction")

    def _calc_factor_a(self, nu):
        """
        Calculate the ratio factor a(Te, ν), which will be used to
        convert the Halpha emission [Rayleigh] to free-free emission
        brightness temperature [K].

        Parameters
        ----------
        nu : float
            The frequency where to calculate the factor a(nu).
            Unit: [MHz]

        Returns
        -------
        a : float
            The factor for Hα to free-free conversion.

        References: [dickinson2003],Eq.(8)
        """
        term1 = 0.183 * nu**0.1 * self.Te**(-0.15)
        term2 = 3.91 - np.log(nu) + 1.5*np.log(self.Te)
        a = term1 * term2
        return a

    def _calc_halpha_to_freefree(self, nu):
        """
        Calculate the conversion factor between Hα emission [Rayleigh]
        to radio free-free emission [K] at frequency ν [MHz].

        Parameters
        ----------
        nu : float
            The frequency where to calculate the conversion factor.
            Unit: [MHz]

        Returns
        -------
        h2f : float
            The conversion factor between Hα emission and free-free emission.

        References: [dickinson2003],Eq.(11)
        NOTE: The above referred formula has a superfluous "10^3" term!
        """
        a = self._calc_factor_a(nu)
        h2f = 38.86 * a * nu**(-2.1) * 10**(290/self.Te) * self.Te**0.667
        return h2f

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
        self._correct_dust_absorption()

        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """
        Simulate the free-free map at the specified frequency.

        Parameters
        ----------
        frequency : float
            The frequency where to simulate the emission map.
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
        ratio_K_R = self._calc_halpha_to_freefree(frequency)
        sky.data = self.halphamap * ratio_K_R
        logger.info("Done simulate map at %.2f [MHz]." % frequency)
        return sky

    def simulate(self, frequencies=None):
        """
        Simulate the emission maps.

        Parameters
        ----------
        frequencies : float, or list[float]
            The frequencies where to simulate the emission map.
            Unit: [MHz]
            Default: None (i.e., use ``self.frequencies``)

        Returns
        -------
        skyfiles : list[str]
            List of the filepath to the written sky files
        """
        if frequencies is None:
            frequencies = self.frequencies
        else:
            frequencies = np.array(frequencies, ndmin=1)

        logger.info("Simulating {name} ...".format(name=self.name))
        skyfiles = []
        for freq in frequencies:
            sky = self.simulate_frequency(freq)
            outfile = self._outfilepath(frequency=freq)
            sky.write(outfile)
            skyfiles.append(outfile)
        logger.info("Done simulate {name}!".format(name=self.name))
        return skyfiles

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("{name}: postprocessing ...".format(name=self.name))
        logger.info("^_^ nothing to do :-)")
