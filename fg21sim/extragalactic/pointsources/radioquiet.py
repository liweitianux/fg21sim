# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .base import BasePointSource
from ...utils import convert


class RadioQuiet(BasePointSource):

    def __init__(self, configs):
        super().__init__(configs)
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsources/num_rq")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/prefix_rq")

    def draw_single_ps(self, freq):
        """
        Designed to draw the radio quiet AGN

        Parameters
        ----------
        ImgMat: np.ndarray
            Two dimensional matrix, to describe the image
        self.ps_catalog: pandas.core.frame.DataFrame
            Data of the point sources
        freq: float
            frequency
        """
        # Init
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        # Gen Tb list
        Tb_list = self.calc_Tb(freq)
        #  Iteratively draw the ps
        num_ps = self.ps_catalog.shape[0]
        for i in range(num_ps):
            # Angle to pix
            lat = (self.ps_catalog['Lat (deg)'] + 90) / 180 * np.pi
            lon = self.ps_catalog['Lon (deg)'] / 180 * np.pi
            pix = hp.ang2pix(self.nside, lat, lon)
            # Gen hpmap
            hpmap[pix] += Tb_list[i]

        return hpmap

    def draw_ps(self, freq):
        """
        Read csv ps list file, and generate the healpix structure vector
        with the respect frequency.
        """
        # Init
        num_freq = len(freq)
        npix = hp.nside2npix(self.nside)
        hpmaps = np.zeros((npix, num_freq))

        # Gen ps_catalog
        self.gen_catalog()
        # get hpmaps
        for i in range(num_freq):
            hpmaps[:, i] = self.draw_single_ps(freq[i])

        return hpmaps

    def calc_single_Tb(self, area, freq):
        """
        Calculate brightness temperatur of a single ps

        Parameters
        ------------
        area: `~astropy.units.Quantity`
              Area of the PS, e.g., `1.0*au.sr2`
        freq: `~astropy.units.Quantity`
              Frequency, e.g., `1.0*au.MHz`

        Return
        ------
        Tb:`~astropy.units.Quantity`
             Average brightness temperature, e.g., `1.0*au.K`
        """
        # Init
        freq_ref = 151 * au.MHz
        freq = freq * au.MHz
        # Refer to Wang et al,'s work listed above.
        I_151 = 10**(np.random.uniform(-4, -3)) * au.Jy
        # Calc flux
        flux = (freq / freq_ref)**(-0.7) * I_151
        # Calc brightness temperature
        Tb = convert.Fnu_to_Tb(flux, area, freq)

        return Tb

    def calc_Tb(self, freq):
        """
        Calculate the surface brightness  temperature of the point sources.

        Parameters
        ------------
        area: `~astropy.units.Quantity`
             Area of the PS, e.g., `1.0*au.sr`
        freq: `~astropy.units.Quantity`
             Frequency, e.g., `1.0*au.MHz`

        Return
        ------
        Tb_list: list
             Point sources brightness temperature
        """
        # Tb_list
        num_ps = self.ps_catalog.shape[0]
        Tb_list = np.zeros((num_ps,))
        # Iteratively calculate Tb
        for i in range(num_ps):
            ps_area = self.ps_catalog['Area (sr)'][i] * au.sr
            Tb_list[i] = self.calc_single_Tb(ps_area, freq).value

        return Tb_list
