# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp

from .flux import Flux
from .base import BasePointSource

class RadioQuiet(BasePointSource):
    def __init__(self,configs):
        super().__init__(configs)
        self._get_configs()

    def _get_configs(self):
        """Load the configs and set the corresponding class attributes"""

        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsources/num_rq")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/prefix_rq")

    def calc_flux(self, freq):
        """
        Calculate the flux and surface brightness of the point source.

        Parameters
        ----------
        ps_type: int
            Type of point source
        freq: float
            frequency
        ps_frame: pandas.core.frame.DataFrame
            Data of the point sources
        """
        # init flux
        ps_flux = Flux(freq, 3)
        # ps_flux_list
        num_ps = self.ps_catelog.shape[0]
        ps_flux_list = np.zeros((num_ps,))
        # Iteratively calculate flux
        for i in range(num_ps):
            ps_area = self.ps_catelog['Area (sr)'][i]
            ps_flux_list[i] = ps_flux.calc_Tb(ps_area)

        return ps_flux_list

    def draw_single_ps(self, freq):
        """
        Designed to draw the radio quiet AGN

        Parameters
        ----------
        ImgMat: np.ndarray
            Two dimensional matrix, to describe the image
        self.ps_catelog: pandas.core.frame.DataFrame
            Data of the point sources
        freq: float
            frequency
        """
        # Init
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        # Gen flux list
        ps_flux_list = self.calc_flux(freq)
        # Angle to pix
        lat = (self.ps_catelog['Lat (deg)'] + 90) / 180 * np.pi
        lon = self.ps_catelog['Lon (deg)'] / 180 * np.pi
        pix = hp.ang2pix(self.nside, lat, lon)
        # Gen hpmap
        hpmap[pix] += ps_flux_list

        return hpmap

    def draw_ps(self):
        """
        Read csv ps list file, and generate the healpix structure vector
        with the respect frequency.
        """
        # Init
        num_freq = len(self.freq)
        npix = hp.nside2npix(self.nside)
        hpmaps = np.zeros((npix,num_freq))

        # Gen ps_catelog
        self.gen_catelog()
        # get hpmaps
        for i in range(num_freq):
            hpmaps[:, i] = self.draw_single_ps(self.freq[i])

        return hpmaps
