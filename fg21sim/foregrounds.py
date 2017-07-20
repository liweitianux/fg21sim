# Copyright (c) 2016-2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Interface to the simulations of various supported foreground components.

Currently supported foregrounds:

- Galactic synchrotron
- Galactic free-free
- Galactic supernova remnants
- Extragalactic clusters of galaxies (radio halos)
- Extragalactic point sources (multiple types)
"""

import os
import logging
from datetime import datetime, timezone
from collections import OrderedDict

import numpy as np
import astropy.units as au
from astropy.io import fits

from .galactic import (Synchrotron as GalacticSynchrotron,
                       FreeFree as GalacticFreeFree,
                       SuperNovaRemnants as GalacticSNR)
from .extragalactic import (GalaxyClusters as EGGalaxyClusters,
                            PointSources as EGPointSources)
from .products import Products
from .sky import get_sky


logger = logging.getLogger(__name__)


class Foregrounds:
    """
    Interface to the simulations of supported foreground components.

    All the enabled components are also combined to make the total foreground
    map, as controlled by the configurations.

    Parameters
    ----------
    configs : `ConfigManager`
        An `ConfigManager` instance containing both the default and user
        configurations.
        For more details, see the example configuration specification.

    Attributes
    ----------
    COMPONENTS_ALL : `OrderedDict`
        Ordered dictionary of all supported simulation components, with keys
        the IDs of the components, and values the corresponding component
        class.
    components_id : list[str]
        List of IDs of the enabled simulation components
    components : `OrderedDict`
        Ordered dictionary of the enabled simulation components, with keys
        the IDs of the components, and values the corresponding component
        instance/object.
    frequencies : 1D `~numpy.ndarray`
        List of frequencies where the foreground components are simulated.
    freq_unit : `~astropy.units.Unit`
        Unit of the frequency
    """
    # All supported foreground components
    COMPONENTS_ALL = OrderedDict([
        ("galactic/synchrotron",       GalacticSynchrotron),
        ("galactic/freefree",          GalacticFreeFree),
        ("galactic/snr",               GalacticSNR),
        ("extragalactic/clusters",     EGGalaxyClusters),
        ("extragalactic/pointsources", EGPointSources),
    ])

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()
        # Initialize the products manifest
        logger.info("Initialize the products manifest ...")
        self.manifestfile = self.configs.get_path("output/manifest")
        if self.manifestfile:
            self.products = Products(self.manifestfile, load=False)
        else:
            self.products = None
            logger.warning("Output products manifest not configured!")
        # Initialize enabled components
        self.components = OrderedDict()
        for comp in self.components_id:
            logger.info("Initialize component: {0}".format(comp))
            comp_cls = self.COMPONENTS_ALL[comp]
            self.components[comp] = comp_cls(configs)
        logger.info("Done initialize %d components!" % len(self.components))

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        self.components_id = self.configs.foregrounds[0]
        logger.info("All supported simulation components: {0}".format(
            ", ".join(list(self.COMPONENTS_ALL.keys()))))
        logger.info("Enabled components: {0}".format(
            ", ".join(self.components_id)))
        #
        self.frequencies = self.configs.frequencies
        self.freq_unit = au.Unit(self.configs.getn("frequency/unit"))
        logger.info("Simulation frequencies: "
                    "{min:.2f} - {max:.2f} {unit} (#{num:d})".format(
                        min=min(self.frequencies),
                        max=max(self.frequencies),
                        num=len(self.frequencies),
                        unit=self.freq_unit))
        #
        self.filename_pattern = self.configs.getn("output/filename_pattern")
        self.use_float = self.configs.getn("output/use_float")
        self.checksum = self.configs.getn("output/checksum")
        self.clobber = self.configs.getn("output/clobber")
        self.combine = self.configs.getn("output/combine")
        self.prefix = self.configs.getn("output/combine_prefix")
        self.output_dir = self.configs.get_path("output/output_dir")

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
        header["COMP"] = ("Combined foreground", "Emission component")
        header.add_comment("COMPONENTS: " + ", ".join(self.components_id))
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

    def _output(self, skymap, frequency):
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
        """Perform the preparation procedures for the final simulations."""
        if self.products:
            self.products.frequencies = (self.frequencies,
                                         str(self.freq_unit))
        logger.info("Perform preprocessing for all enabled components ...")
        for comp_obj in self.components.values():
            comp_obj.preprocess()

    def simulate(self):
        """
        Simulate the enabled components, as well as combine all the
        simulated components to make up the total foregrounds.

        This is the *main interface* to the foreground simulations.

        NOTE
        ----
        For each requested frequency, all enabled components are simulated,
        which are combined to compose the total foreground and save to disk.
        In this way, less memory is required, since the number of components
        are generally much less than the number of frequency bins.
        """
        sky = get_sky(configs=self.configs)
        nfreq = len(self.frequencies)
        for freq_id, freq in enumerate(self.frequencies):
            logger.info("[#{0}/{1}] ".format(freq_id+1, nfreq) +
                        "Simulating components at {freq} {unit} ...".format(
                            freq=freq, unit=self.freq_unit))
            if self.combine:
                skymap_comb = np.zeros(shape=sky.shape)
            for comp_id, comp_obj in self.components.items():
                skymap, filepath = comp_obj.simulate_frequency(freq)
                if filepath and self.products:
                    self.products.add_product(comp_id, freq_id, filepath)
                if self.combine:
                    skymap_comb += skymap
            if self.combine:
                filepath_comb = self._output(skymap_comb, freq)
                if self.products:
                    self.products.add_product("combined", freq_id,
                                              filepath_comb)

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        logger.info("Perform postprocessing for all enabled components ...")
        for comp_obj in self.components.values():
            comp_obj.postprocess()
        # Save the products manifest
        if self.products:
            self.products.dump(clobber=self.clobber, backup=True)
