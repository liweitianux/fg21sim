# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

from .starforming import StarForming
from .starbursting import StarBursting
from .radioquiet import RadioQuiet
from .fr1 import FRI
from .fr2 import FRII


class PointSources:
    """
    This class namely pointsource is designed to generate PS catelogs,
    read csv format PS lists, calculate the flux and surface brightness
    of the sources at different frequencies, and then ouput hpmaps

    functions
    ---------
    read_csv
        read the csv format files, judge the PS type and
        transformed to be iterable numpy.ndarray.

    calc_flux
        calculate the flux and surface brightness of the PS.

    draw_elp
        processing on the elliptical and circular core or lobes.

    draw_circle
        processing on the circular star forming or bursting galaxies

    draw_ps
        generate hpmap with respect the imput PS catelog
    """

    def __init__(self, configs):
        self.configs = configs
        self._get_configs()
        self.files = []
        self.ps = []

    def _get_configs(self):
        """Load configs and set the attributes"""
        # nside of the healpix cell
        self.nside = self.configs.getn("common/nside")
        # frequencies
        self.freq = self.configs.getn("frequency/frequencies")
        # save flag
        self.save = self.configs.getn("extragalactic/pointsources/save")

    def get_ps(self):
        """Generate the catelogs"""
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

        # Save
        if self.save:
            self.sf.save_as_csv()
            self.sb.save_as_csv()
            self.rq.save_as_csv()
            self.fr1.save_as_csv()
            self.fr2.save_as_csv()

    def get_hpmaps(self):
        """Get hpmaps"""
        hpmaps = (self.sf.draw_ps() + self.sb.draw_ps() +
                  self.rq.draw_ps() + self.fr1.draw_ps() +
                  self.fr2.draw_ps())

        return hpmaps
