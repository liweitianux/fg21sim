# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Extragalactic point sources (ps) simulation
"""

import logging
import numpy as np

from .starforming import StarForming
from .starbursting import StarBursting
from .radioquiet import RadioQuiet
from .fr1 import FRI
from .fr2 import FRII


logger = logging.getLogger(__name__)


class PointSources:
    """
    This class namely pointsource is designed to generate PS catelogs,
    read csv format PS lists, calculate the flux and surface brightness
    of the sources at different frequencies, and then ouput hpmaps

    Parameters
    ----------
    configs: ConfigManager object
        An 'ConfigManager' object contains default and user configurations.
        For more details, see the example config specification.

    Functions
    ---------
    get_ps
        Generate the ps catelogs for each type.

    simulate
        Simulate and project pss to the healpix map.
    """

    def __init__(self, configs):
        self.configs = configs
        self._get_configs()

    def _get_configs(self):
        """Load configs and set the attributes"""
        # nside of the healpix cell
        self.nside = self.configs.getn("common/nside")
        # save flag
        self.save = self.configs.getn("extragalactic/pointsources/save")

    def preprocess(self):
        """Preprocess and generate the catelogs"""
        logger.info("Generating PS catelogs...")
        # Init
        self.sf = StarForming(self.configs)
        self.sb = StarBursting(self.configs)
        self.rq = RadioQuiet(self.configs)
        self.fr1 = FRI(self.configs)
        self.fr2 = FRII(self.configs)

        # Gen ps_catelog
        self.sf.gen_catelog()
        self.sb.gen_catelog()
        self.rq.gen_catelog()
        self.fr1.gen_catelog()
        self.fr2.gen_catelog()
        logger.info("Generating PS catelogs done!")


    def simulate_frequency(self,freq):
        """Simulate the point sources and output hpmaps"""
        # Projecting
        logger.info("Generating PS hpmaps...")
        hpmap_f = (self.sf.draw_single_ps(freq) +
                  self.sb.draw_single_ps(freq) +
                  self.rq.draw_single_ps(freq) +
                  self.fr1.draw_single_ps(freq) +
                  self.fr2.draw_single_ps(freq))
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
        logger.info("Saving simulated catelogs...")
        # Save the catalog actually used in the simulation
        if self.save:
            self.sf.save_as_csv()
            self.sb.save_as_csv()
            self.rq.save_as_csv()
            self.fr1.save_as_csv()
            self.fr2.save_as_csv()
