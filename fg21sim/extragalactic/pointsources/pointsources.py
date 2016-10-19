# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Extragalactic point sources (ps) simulation
"""

import logging
import numpy as np
import healpy as hp
from collections import OrderedDict

from .starforming import StarForming
from .starbursting import StarBursting
from .radioquiet import RadioQuiet
from .fr1 import FRI
from .fr2 import FRII


logger = logging.getLogger(__name__)


class PointSources:
    """
    This class namely pointsource is designed to generate PS catalogs,
    read csv format PS lists, calculate the flux and surface brightness
    of the sources at different frequencies, and then ouput hpmaps

    Parameters
    ----------
    configs: ConfigManager object
        An 'ConfigManager' object contains default and user configurations.
        For more details, see the example config specification.

    Functions
    ---------
    preprocessing
        Generate the ps catalogs for each type.
        simulate_frequency
            Simualte point sources at provivded frequency
    simulate
        Simulate and project PSs to the healpix map.
        postprocessing
            Save catalogs
    """
    PSCOMPONENTS_ALL = OrderedDict([
        ("starforming", StarForming),
        ("starbursting", StarBursting),
        ("radioquiet", RadioQuiet),
        ("FRI", FRI),
        ("FRII", FRII),
    ])

    def __init__(self, configs):
        self.configs = configs
        self._set_configs()
        self.pscomps = OrderedDict()
        for comp in self.pscomps_id:
            logger.info("Initlalize PS component: {0}".format(comp))
            comp_type = self.PSCOMPONENTS_ALL[comp]
            self.pscomps[comp] = comp_type(configs)
        logger.info("Done initlalize %d PS components!" %
                    len(self.pscomps))

    def _set_configs(self):
        """Load configs and set the attributes"""
        # Prefix of simulated point sources
        self.pscomps_id = self.configs.getn("extragalactic/pscomponents")
        if self.pscomps_id is None:
            self.pscomps_id = ['starforming', 'starbursting', 'radioquiet',
                               'FRI', 'FRII']
        print(self.pscomps_id)
        # nside of the healpix cell
        self.nside = self.configs.getn("common/nside")
        # save flag
        self.save = self.configs.getn("extragalactic/pointsources/save")

    def preprocess(self):
        """Preprocess and generate the catalogs"""
        logger.info("Generating PS catalogs...")
        # Gen ps_catalog
        for pscomp_obj in self.pscomps.values():
            pscomp_obj.gen_catalog()
        logger.info("Generating PS catalogs done!")

    def simulate_frequency(self, freq):
        """Simulate the point sources and output hpmaps"""
        npix = hp.nside2npix(self.nside)
        hpmap_f = np.zeros((npix,))
        # Projecting
        logger.info("Generating PS hpmaps...")
        for pscomp_obj in self.pscomps.values():
            hpmap_f += pscomp_obj.draw_single_ps(freq)
        logger.info("Generating PS hpmaps done!")

        return hpmap_f

    def simulate(self, frequencies):
        """Simulate the emission (HEALPix) maps of all Galactic SNRs for
        every specified frequency.

        Parameters
        ----------
        frequency : list[float]
            List of frequencies (unit: `self.freq_unit`) where the
            simulation performed.

        Returns
        -------
        hpmaps : list[1D `~numpy.ndarray`]
            List of HEALPix maps (in RING ordering) at each frequency.
        """
        hpmaps = []
        for f in np.array(frequencies, ndmin=1):
            hpmap_f = self.simulate_frequency(f)
            hpmaps.append(hpmap_f)
        return hpmaps

    def postprocess(self):
        """Perform the post-simulation operations before the end."""
        # Save the catalog actually used in the simulation
        if self.save:
            logger.info("Saving simulated catalogs...")
            for pscomp_obj in self.pscomps.values():
                pscomp_obj.save_as_csv()
            logger.info("Saving simulated catalogs done!")
