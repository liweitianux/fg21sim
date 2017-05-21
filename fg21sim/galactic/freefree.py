# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Diffuse Galactic free-free emission simulations.
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits
import astropy.units as au

from ..sky import get_sky


logger = logging.getLogger(__name__)


class FreeFree:
    """
    Simulate the diffuse Galactic free-free emission.

    The [Dickinson2003]_ method is followed to derive the free-free template.
    The H\alpha survey map [Finkbeiner2003]_ is first corrected for dust
    absorption using the infrared 100-\mu{}m dust map [Schlegel1998]_,
    and then converted to free-free emission map (brightness temperature).

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
    .. [Dickinson2003]
       Dickinson, C.; Davies, R. D.; Davis, R. J.,
       "Towards a free-free template for CMB foregrounds",
       2003, MNRAS, 341, 369,
       http://adsabs.harvard.edu/abs/2003MNRAS.341..369D

    .. [Finkbeiner2003]
       Finkbeiner, Douglas P.,
       "A Full-Sky Hα Template for Microwave Foreground Prediction",
       2003, ApJS, 146, 407,
       http://adsabs.harvard.edu/abs/2003ApJS..146..407F

    .. [Schlegel1998]
       Schlegel, David J.; Finkbeiner, Douglas P.; Davis, Marc,
       "Maps of Dust Infrared Emission for Use in Estimation of Reddening
       and Cosmic Microwave Background Radiation Foregrounds",
       1998, ApJ, 500, 525,
       http://adsabs.harvard.edu/abs/1998ApJ...500..525S
    """
    # Component name
    name = "Galactic free-free"

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        comp = "galactic/freefree"
        self.halphamap_path = self.configs.get_path(comp+"/halphamap")
        self.halphamap_unit = au.Unit(
            self.configs.getn(comp+"/halphamap_unit"))
        self.dustmap_path = self.configs.get_path(comp+"/dustmap")
        self.dustmap_unit = au.Unit(
            self.configs.getn(comp+"/dustmap_unit"))
        self.prefix = self.configs.getn(comp+"/prefix")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        #
        logger.info("Loaded and set up configurations")

    def _load_maps(self):
        """
        Load the H{\alpha} map, and dust map.
        """
        sky = get_sky(self.configs)
        logger.info("Loading H[alpha] map ...")
        self.halphamap = sky.load(self.halphamap_path)
        # TODO: Validate & convert unit
        if self.halphamap_unit != au.Unit("Rayleigh"):
            raise ValueError("unsupported Halpha map unit: {0}".format(
                self.halphamap_unit))
        logger.info("Loading dust map ...")
        self.dustmap = sky.load(self.dustmap_path)
        # TODO: Validate & convert unit
        if self.dustmap_unit != au.Unit("MJy / sr"):
            raise ValueError("unsupported dust map unit: {0}".format(
                self.dustmap_unit))

    def _correct_dust_absorption(self):
        """
        Correct the H{\alpha} map for dust absorption using the
        100-{\mu}m dust map.

        References: [Dickinson2003]: Eq.(1, 3); Sec.(2.5)
        """
        if hasattr(self, "_dust_corrected") and self._dust_corrected:
            return
        #
        logger.info("Correct H[alpha] map for dust absorption")
        # Effective dust fraction in the LoS actually absorbing Halpha
        f_dust = 0.33
        logger.info("Effective dust fraction: {0}".format(f_dust))
        # NOTE:
        # Mask the regions where the true Halpha absorption is uncertain.
        # When the dust absorption goes rather large, the true Halpha
        # absorption can not well determined.
        # Therefore, the regions where the calculated Halpha absorption
        # greater than 1.0 mag are masked out.
        halpha_abs_th = 1.0  # Halpha absorption threshold, unit: [ mag ]
        # Corresponding dust absorption threshold, unit: [ MJy / sr ]
        dust_abs_th = halpha_abs_th / 0.0462 / f_dust
        logger.info("Dust absorption mask threshold: " +
                    "{0:.1f} MJy/sr ".format(dust_abs_th) +
                    "<-> H[alpha] absorption threshold: " +
                    "{0:.1f} mag".format(halpha_abs_th))
        mask = (self.dustmap.data > dust_abs_th)
        self.dustmap.data[mask] = np.nan
        fp_mask = 100 * mask.sum() / self.dustmap.data.size
        logger.warning("Dust map masked fraction: {0:.1f}%".format(fp_mask))
        #
        halphamap_corr = (self.halphamap.data *
                          10**(self.dustmap.data * 0.0185 * f_dust))
        self.halphamap.data = halphamap_corr
        self._dust_corrected = True
        logger.info("Done dust absorption correction")

    def _calc_ratio_a(self, Te, nu_GHz):
        """Calculate the ratio factor a(T, nu), which will be used to
        transform the Halpha emission (Rayleigh) to free-free emission
        brightness temperature (mK).

        References: [Dickinson2003], Eq.(8)
        """
        term1 = 0.366 * nu_GHz**0.1 * Te**(-0.15)
        term2 = np.log(4.995e-2 / nu_GHz) + 1.5*np.log(Te)
        a = term1 * term2
        return a

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
        header["COMP"] = ("Galactic free-free emission",
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

    def output(self, skymap, frequency):
        """
        Write the simulated free-free map to disk with proper header
        keywords and history.

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
        # Correct for dust absorption
        self._correct_dust_absorption()
        #
        self._preprocessed = True

    def simulate_frequency(self, frequency):
        """
        Simulate the free-free map at the specified frequency.

        References: [Dickinson2003], Eq.(11)

        NOTE: [Dickinson2003], Eq.(11) may wrongly have the "10^3" term.

        Returns
        -------
        hpmap_f : 1D `~numpy.ndarray`
            The HEALPix map (RING ordering) at the input frequency.
        filepath : str
            The (absolute) path to the output HEALPix file if saved,
            otherwise ``None``.
        """
        self.preprocess()
        #
        logger.info("Simulating {name} map at {freq} ({unit}) ...".format(
            name=self.name, freq=frequency, unit=self.freq_unit))
        # Assumed electron temperature [ K ]
        Te = 7000.0
        T4 = Te / 1e4
        nu = frequency * self.freq_unit.to(au.GHz)  # frequency [ GHz ]
        ratio_a = self._calc_ratio_a(Te, nu)
        # NOTE: ignored the "10^3" term in the referred equation
        ratio_mK_R = (8.396 * ratio_a * nu**(-2.1) *
                      T4**0.667 * 10**(0.029/T4) * (1+0.08))
        # Use "Kelvin" as the brightness temperature unit
        ratio_K_R = ratio_mK_R * au.mK.to(au.K)
        skymap_f = self.halphamap.data * ratio_K_R
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
        hpmaps : list[1D `~numpy.ndarray`]
            List of HEALPix maps (in RING ordering) at each frequency.
        paths : list[str]
            List of (absolute) path to the output HEALPix maps.
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
