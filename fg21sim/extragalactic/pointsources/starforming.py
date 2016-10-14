# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource


class StarForming(BasePointSource):
    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.columns.append('radius (rad)')
        self.nCols = len(self.columns)
        self._get_configs()

    def _get_configs(self):
        """ Load the configs and set the corresponding class attributes"""
        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsources/num_sf")
        # Luminosity at 1.4GHz
        self.lumo_1400 = self.configs.getn(
            "extragalactic/pointsources/lumo_1400")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/prefix_sf")

    def get_radius(self):
        Temp = (0.22 * np.log10(self.lumo_1400) -
                np.log10(1 + self.z) - 3.32)
        self.radius = 10 ** Temp / 2 * au.Mpc

        return self.radius

    def gen_single_ps(self):
        """
        Generate single point source, and return its data as a list.
        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA
        self.radius = self.param.get_angle(self.get_radius()) # [rad]
        # Area
        self.area = np.pi * self.radius**2  #[sr] ?
        # Position
        x = np.random.uniform(0,1)
        self.theta = np.arccos(x)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

        ps_list = [self.z, self.dA.value, self.theta.value,
             self.phi.value, self.area.value, self.radius.value]

        return ps_list
