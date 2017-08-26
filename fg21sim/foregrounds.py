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

import logging
from collections import OrderedDict

from .galactic import (Synchrotron as GalacticSynchrotron,
                       FreeFree as GalacticFreeFree,
                       SuperNovaRemnants as GalacticSNR)
from .extragalactic import (GalaxyClusters as EGGalaxyClusters,
                            PointSources as EGPointSources)
from .products import Products


logger = logging.getLogger(__name__)


class Foregrounds:
    """
    Interface to the simulations of supported foreground components.

    Parameters
    ----------
    configs : `~ConfigManager`
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
        """
        Load the configs and set the corresponding class attributes.
        """
        self.components_id = self.configs.foregrounds[0]
        logger.info("All supported simulation components: {0}".format(
            ", ".join(list(self.COMPONENTS_ALL.keys()))))
        logger.info("Enabled components: {0}".format(
            ", ".join(self.components_id)))
        #
        self.frequencies = self.configs.frequencies
        logger.info("Simulation frequencies: "
                    "{min:.2f} - {max:.2f} [MHz] (#{num:d})".format(
                        min=min(self.frequencies),
                        max=max(self.frequencies),
                        num=len(self.frequencies)))
        #
        self.clobber = self.configs.getn("output/clobber")

    def preprocess(self):
        """
        Perform the preparation procedures for the final simulations.
        """
        if self.products:
            self.products.frequencies = (self.frequencies, "MHz")
        logger.info("Perform preprocessing for all enabled components ...")
        for comp_obj in self.components.values():
            comp_obj.preprocess()

    def simulate(self):
        """
        Simulate the enabled components.
        """
        nfreq = len(self.frequencies)
        for freq_id, freq in enumerate(self.frequencies):
            logger.info("[#{0}/{1}] ".format(freq_id+1, nfreq) +
                        "Simulating components at %.2f [MHz] ..." % freq)
            for comp_id, comp_obj in self.components.items():
                skymap, filepath = comp_obj.simulate_frequency(freq)
                if filepath and self.products:
                    self.products.add_product(comp_id, freq_id, filepath)

    def postprocess(self):
        """
        Perform the post-simulation operations before the end.
        """
        logger.info("Perform postprocessing for all enabled components ...")
        for comp_obj in self.components.values():
            comp_obj.postprocess()
        # Save the products manifest
        if self.products:
            self.products.dump(clobber=self.clobber, backup=True)
