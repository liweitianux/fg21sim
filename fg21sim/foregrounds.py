# Copyright (c) 2016-2019 Weitian LI <wt@liwt.net>
# MIT License

"""
Interface to the simulations of various supported foreground components.

Currently supported foregrounds:

- Galactic synchrotron
- Galactic free-free
- Galactic supernova remnants
- Extragalactic clusters of galaxies (radio halos)
"""

import logging
import time
from collections import OrderedDict

from .galactic import (Synchrotron as GalacticSynchrotron,
                       FreeFree as GalacticFreeFree,
                       SuperNovaRemnants as GalacticSNR)
from .extragalactic import GalaxyClusters as EGGalaxyClusters
from .products import Products


logger = logging.getLogger(__name__)

# All supported foreground components:
COMPONENTS_ALL = OrderedDict([
    ("galactic/synchrotron",       GalacticSynchrotron),
    ("galactic/freefree",          GalacticFreeFree),
    ("galactic/snr",               GalacticSNR),
    ("extragalactic/clusters",     EGGalaxyClusters),
])


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
    componentsID : list[str]
        List of IDs of the enabled simulation components
    frequencies : 1D `~numpy.ndarray`
        List of frequencies where the foreground components are simulated.
    """
    def __init__(self, configs):
        self.configs = configs
        self._set_configs()

        logger.info("Initialize the products manifest ...")
        self.manifestfile = self.configs.get_path("output/manifest")
        if self.manifestfile:
            self.products = Products(self.manifestfile, load=False)
            self.products.frequencies = (self.frequencies, "MHz")
        else:
            self.products = None
            logger.warning("Output products manifest not configured!")

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp_enabled, comp_available = self.configs.foregrounds
        self.componentsID = comp_enabled
        logger.info("All supported simulation components: {0}".format(
            ", ".join(comp_available)))
        logger.info("Enabled components: {0}".format(
            ", ".join(self.componentsID)))

        self.frequencies = self.configs.frequencies
        logger.info("Simulation frequencies: "
                    "{min:.2f} - {max:.2f} [MHz] (#{num:d})".format(
                        min=min(self.frequencies),
                        max=max(self.frequencies),
                        num=len(self.frequencies)))

        self.clobber = self.configs.getn("output/clobber")

    def preprocess(self):
        """
        Perform the global preparation procedures.
        """
        logger.info("Global preprocessing ... (empty)")

    def simulate_component(self, compID):
        """
        Simulate the specified foreground component.
        """
        logger.info("==================================================")
        logger.info(">>> Simulate component: %s <<<" % compID)
        logger.info("==================================================")
        t_start = time.perf_counter()

        comp_cls = COMPONENTS_ALL[compID]
        comp_obj = comp_cls(self.configs)
        comp_obj.preprocess()
        skyfiles = comp_obj.simulate()
        if self.products and skyfiles:
            self.products.add_component(compID, skyfiles)
        comp_obj.postprocess()

        t_cost = time.perf_counter() - t_start
        logger.info("--------------------------------------------------")
        if t_cost <= 3*60:
            logger.info("Elapsed time: %.1f [sec]" % t_cost)
        else:
            logger.info("Elapsed time: %.1f [min]" % (t_cost/60))
        logger.info("--------------------------------------------------")

    def simulate(self):
        """
        Do simulation for all enabled components.
        """
        times = []
        for compID in self.componentsID:
            t1 = time.perf_counter()
            self.simulate_component(compID)
            t2 = time.perf_counter()
            times.append((compID, t1, t2))

        logger.info("==================================================")
        logger.info(">>> Time usage <<<")
        logger.info("==================================================")
        for compID, t1, t2 in times:
            t_cost = t2 - t1
            if t_cost <= 3*60:
                logger.info("%s : %.1f [sec]" % (compID, t_cost))
            else:
                logger.info("%s : %.1f [min]" % (compID, t_cost/60))
        logger.info("--------------------------------------------------")

    def postprocess(self):
        """
        Perform the global post-simulation operations.
        """
        logger.info("Global postprocessing ...")
        if self.products:
            logger.info("Save the products manifest ...")
            self.products.dump(clobber=self.clobber, backup=True)
