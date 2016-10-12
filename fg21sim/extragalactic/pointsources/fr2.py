# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import astropy.units as au
from .base import BasePointSource
from .psparams import PixelParams

class FRII(BasePointSource):
    """
    Generate Faranoff-Riley II (FRII) AGN

    Parameters
    ----------
    lobe_maj: float
        The major half axis of the lobe
    lobe_min: float
        The minor half axis of the lobe
    lobe_ang: float
        The rotation angle of the lobe from LOS
    """
    def __init__(self,configs):
        super().__init__(configs)
        self.columns.extend(
            ['lobe_maj (rad)', 'lobe_min (rad)', 'lobe_ang (deg)'])
        self.nCols += 3
        self._get_configs()

    def _get_configs(self):
        """ Load the configs and set the corresponding class attributes"""
        # point sources amount
        self.NumPS = self.configs.getn("extragalactic/pointsource/num_fr2")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsource/prefix_fr2")

    def gen_lobe(self):
        """
        According to Wang's work, the linear scale at redshift z
        obeys to U(0,D0(1+z)^(-1.4))
        """
        D0 = 1
        self.lobe_maj = 0.5 * np.random.uniform(0,
            D0 * (1 + self.z)**(-1.4)) * au.Mpc
        self.lobe_min = self.lobe_maj * np.random.uniform(0.2, 1) * au.Mpc

        # Different from FRI
        self.lobe_ang = np.random.uniform(0, np.pi / 3)/np.pi * 180 * au.deg

        # Transform to pixel
        self.lobe_maj = self.param.get_angle(self.lobe_maj)
        self.lobe_min = self.param.get_angle(self.lobe_min)
        lobe = [self.lobe_maj.value, self.lobe_min.value, self.lobe_ang.value]

        return lobe

    def gen_single_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA

        # Position
        self.theta = np.random.uniform(0,np.pi)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

        # lobe
        lobe = self.gen_lobe()

        # Area
        self.area = np.pi * self.lobe_maj * self.lobe_min

        ps_list = [self.z, self.dA.value, self.theta.value,
            self.phi.value, self.area.value]
        ps_list.extend(lobe)

        return ps_list
