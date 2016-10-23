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
        # Number density matrix
        self.rho_mat = self.calc_number_density()
        # Cumulative distribution of z and lumo
        self.cdf_z, self.cdf_lumo = self.calc_cdf()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        # point sources amount
        self.num_ps = self.configs.getn(
        "extragalactic/pointsources/radioquiet/numps")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/radioquiet/prefix")
        # redshift bin
        z_type = self.configs.getn(
            "extragalactic/pointsources/radioquiet/z_type")
        if z_type == 'custom':
            start = self.configs.getn(
               "extragalactic/pointsources/radioquiet/z_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/radioquiet/z_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/radioquiet/z_step")
            self.zbin = np.arange(start,stop+step,step)
        else:
            self.zbin = np.arange(0.1,10,0.1);
        # luminosity bin
        lumo_type = self.configs.getn(
            "extragalactic/pointsources/radioquiet/lumo_type")
        if lumo_type == 'custom':
            start = self.configs.getn(
                "extragalactic/pointsources/radioquiet/lumo_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/radioquiet/lumo_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/radioquiet/lumo_step")
            self.lumobin = np.arange(start,stop+step,step)
        else:
            self.lumobin = np.arange(18.7,25.7,0.1); # [W/Hz/sr]

    def calc_number_density(self):
        """
        Calculate number density rho(lumo,z) of FRI

        References
        ------------
        [1] Wilman et al.,
            "A semi-empirical simulation of the extragalactic radio continuum
             sky for next generation radio telescopes",
             2008, MNRAS, 388, 1335-1348.
             http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W

        Returns
        -------
        rho_mat: np.ndarray
            Number density matris (joint-distribution of luminosity and
        reshift).
        """
        # Init
        rho_mat = np.zeros((len(self.lumobin),len(self.zbin)))
        # Parameters
        # Refer to Willman's section 2.4
        alpha = 0.7 # spectral index
        lumo_star = 10.0**21.3 # critical luminosity at 1400MHz
        rho_l0 = 10.0**(-7) # normalization constant
        z1 = 1.9 # cut-off redshift
        k1 = -3.27 # index of space density revolution
        # Calculation
        for i, z in enumerate(self.zbin):
            if z <= z1:
                rho_mat[:,i] = ((rho_l0 * (10**self.lumobin/lumo_star)**
                                 -alpha * np.exp(-10**self.lumobin /
                                                 lumo_star)) * (1+z)**k1)
            else:
                rho_mat[:,i] = ((rho_l0 * (10**self.lumobin/lumo_star)**
                                 -alpha * np.exp(-10**self.lumobin /
                                                 lumo_star)) * (1+z1)**k1)
        return rho_mat

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
        freq_ref = 1400 * au.MHz
        freq = freq * au.MHz
        # Luminosity at 1400MHz
        lumo_1400 = self.lumo.to(au.Jy)  # [W/Hz/Sr to Jy]
        # Calc flux
        flux = (freq / freq_ref)**(-0.7) * lumo_1400
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
