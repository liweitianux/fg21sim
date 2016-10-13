# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource


class StarBursting(BasePointSource):

    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    # Init
    def __init__(self, configs):
        super().__init__(configs)
        self.columns.append('radius (rad)')
        self.nCols = len(self.columns)
        self._get_configs()

    def _get_configs(self):
        """Load the configs and set the corresponding class attributes"""
        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsource/num_sb")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsource/prefix_sb")

    def get_radius(self):
        if self.z <= 1.5:
            self.radius = (1 + self.z)**2.5 * 1e-3 * au.Mpc
        else:
            self.radius = 10 * 1e-3 * au.Mpc

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
        self.radius = self.param.get_angle(self.get_radius())
        # Area
        self.area = np.pi * self.radius**2 #[sr]
        # Position
        x = np.random.uniform(0,1)
        self.theta = np.arccos(x)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

        ps_list = [self.z, self.dA.value, self.theta.value,
                   self.phi.value, self.area.value, self.radius.value]

        return ps_list
