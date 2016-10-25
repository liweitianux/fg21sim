# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource
from ...utils import grid
from ...utils import convert


class FRI(BasePointSource):
    """
    Generate Faranoff-Riley I (FRI) AGN

    Parameters
    ----------
    lobe_maj: float
        The major half axis of the lobe
    lobe_min: float
        The minor half axis of the lobe
    lobe_ang: float
        The rotation angle of the lobe with respect to line of sight

    Reference
    ----------
    [1] Wang J et al.,
        "How to Identify and Separate Bright Galaxy Clusters from the
        Low-frequency Radio Sky?",
        2010, ApJ, 723, 620-633.
        http://adsabs.harvard.edu/abs/2010ApJ...723..620W
    [2] Fast cirles drawing
        https://github.com/liweitianux/fg21sim/fg21sim/utils/draw.py
        https://github.com/liweitianux/fg21sim/fg21sim/utils/grid.py
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.columns.extend(
            ['lobe_maj (rad)', 'lobe_min (rad)', 'lobe_ang (deg)'])
        self.nCols = len(self.columns)
        self._set_configs()
        # Paramters for core/lobe ratio
        # Willman et al. 2008 Sec2.5.(iii)-(iv)
        self.xmed = -2.6
        # Lorentz factor of the jet
        self.gamma = 6
        # Number density matrix
        self.rho_mat = self.calc_number_density()
        # Cumulative distribution of z and lumo
        self.cdf_z, self.cdf_lumo = self.calc_cdf()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        # point sources amount
        self.num_ps = self.configs.getn(
            "extragalactic/pointsources/FRI/numps")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/FRI/prefix")
        # redshift bin
        z_type = self.configs.getn(
            "extragalactic/pointsources/FRI/z_type")
        if z_type == 'custom':
            start = self.configs.getn(
                "extragalactic/pointsources/FRI/z_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/FRI/z_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/FRI/z_step")
            self.zbin = np.arange(start, stop + step, step)
        else:
            self.zbin = np.arange(0.1, 10, 0.05)
        # luminosity bin
        lumo_type = self.configs.getn(
            "extragalactic/pointsources/FRI/lumo_type")
        if lumo_type == 'custom':
            start = self.configs.getn(
                "extragalactic/pointsources/FRI/lumo_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/FRI/lumo_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/FRI/lumo_step")
            self.lumobin = np.arange(start, stop + step, step)
        else:
            self.lumobin = np.arange(20, 28, 0.1)  # [W/Hz/sr]

    def calc_number_density(self):
        """
        Calculate number density rho(lumo,z) of FRI

        References
        ----------
        [1] Wilman et al.,
             "A semi-empirical simulation of the extragalactic radio continuum
             sky for next generation radio telescopes",
             2008, MNRAS, 388, 1335-1348.
             http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W
        [2] Willott et al.,
             "The radio luminosity function from the low-frequency 3CRR,
             6CE and 7CRS complete samples",
             2001, MNRAS, 322, 536-552.
             http://adsabs.harvard.edu/abs/2001MNRAS.322..536W

        Returns
        -------
        rho_mat: np.ndarray
            Number density matris (joint-distribution of luminosity and
            reshift).
        """
        # Init
        rho_mat = np.zeros((len(self.lumobin), len(self.zbin)))
        # Parameters
        # Refer to [2] Table. 1  model C and Willman's section 2.4
        alpha = 0.539  # spectral index
        lumo_star = 10.0**26.1  # critical luminosity
        rho_l0 = 10.0**(-7.120)  # normalization constant
        z1 = 0.706  # cut-off redshift
        z2 = 2.5  # cut-off redshift adviced by Willman
        k1 = 4.30  # index of space density revolution
        # Calculation
        for i, z in enumerate(self.zbin):
            if z <= z1:
                rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                                  (-alpha) *
                                  np.exp(-10**self.lumobin / lumo_star)) *
                                 (1 + z)**k1)
            elif z <= z2:
                rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                                  (-alpha) *
                                  np.exp(-10**self.lumobin / lumo_star)) *
                                 (1 + z1)**k1)
            else:
                rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                                  (-alpha) *
                                  np.exp(-10**self.lumobin / lumo_star)) *
                                 (1 + z)**-z2)

        return rho_mat

    def gen_lobe(self):
        D0 = 1 * au.Mpc
        self.lobe_maj = 0.5 * np.random.uniform(
            0, D0.value * (1 + self.z)**(-1.4)) * au.Mpc
        self.lobe_min = self.lobe_maj * np.random.uniform(0.2, 1) * au.Mpc
        self.lobe_ang = np.random.uniform(0, np.pi) / np.pi * 180 * au.deg

        # Transform to pixel
        self.lobe_maj = self.param.get_angle(self.lobe_maj)
        self.lobe_min = self.param.get_angle(self.lobe_min)
        lobe = [self.lobe_maj.value, self.lobe_min.value,
                self.lobe_ang.value]

        return lobe

    def gen_single_ps(self):
        """
        Generate single point source, and return its data as a list.
        """
        # Redshift and luminosity
        self.z, self.lumo = self.get_lumo_redshift()
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA
        # W/Hz/Sr to Jy
        self.lumo = self.lumo / \
            self.dA.to(au.m).value**2 * au.W / au.Hz / au.m / au.m
        self.lumo = self.lumo.to(au.Jy)
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi * 180 - 90) * au.deg
        self.lon = np.random.uniform(0, np.pi * 2) / np.pi * 180 * au.deg

        # lobe
        lobe = self.gen_lobe()

        # Area
        self.area = np.pi * self.lobe_maj * self.lobe_min

        ps_list = [self.z, self.dA.value, self.lumo.value, self.lat.value,
                   self.lon.value, self.area.value]

        ps_list.extend(lobe)

        return ps_list

    def draw_single_ps(self, freq):
        """
        Designed to draw the elliptical lobes of FRI and FRII

        Prameters
        ---------
        nside: int and dyadic
        self.ps_catalog: pandas.core.frame.DataFrame
            Data of the point sources
        ps_type: int
            Class type of the point soruces
        freq: float
            frequency
        """
        # Init
        resolution = 0.001  # [degree]
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        num_ps = self.ps_catalog.shape[0]
        # Gen flux list
        Tb_list = self.calc_Tb(freq)
        ps_lobe = Tb_list[:, 1]
        # Iteratively draw ps
        for i in range(num_ps):
            # Parameters
            c_lat = self.ps_catalog['Lat (deg)'][i]  # core lat [au.deg]
            c_lon = self.ps_catalog['Lon (deg)'][i]  # core lon [au.deg]
            lobe_maj = self.ps_catalog['lobe_maj (rad)'][i] * au.rad
            lobe_min = self.ps_catalog['lobe_min (rad)'][i] * au.rad
            lobe_ang = self.ps_catalog['lobe_ang (deg)'][i] / 180 * np.pi

            # Lobe1
            lobe1_lat = (lobe_maj / 2).to(au.deg) * np.cos(lobe_ang)
            lobe1_lat = c_lat + lobe1_lat.value
            lobe1_lon = (lobe_maj / 2).to(au.deg) * np.sin(lobe_ang)
            lobe1_lon = c_lon + lobe1_lon.value
            # draw
            # Fill with ellipse
            lon, lat, gridmap = grid.make_grid_ellipse(
                (lobe1_lon, lobe1_lat),
                (lobe_maj.to(au.deg).value, lobe_min.to(au.deg).value),
                resolution, lobe_ang / np.pi * 180)
            indices, values = grid.map_grid_to_healpix(
                (lon, lat, gridmap), self.nside)
            hpmap[indices] += ps_lobe[i]

            # Lobe2
            lobe2_lat = (lobe_maj / 2).to(au.deg) * np.cos(lobe_ang + np.pi)
            lobe2_lat = c_lat + lobe2_lat.value
            lobe2_lon = (lobe_maj / 2).to(au.deg) * np.sin(lobe_ang + np.pi)
            lobe2_lon = c_lon + lobe2_lon.value
            # draw
            # Fill with ellipse
            lon, lat, gridmap = grid.make_grid_ellipse(
                (lobe2_lon, lobe2_lat),
                (lobe_maj.to(au.deg).value, lobe_min.to(au.deg).value),
                resolution, lobe_ang / np.pi * 180)
            indices, values = grid.map_grid_to_healpix(
                (lon, lat, gridmap), self.nside)
            hpmap[indices] += ps_lobe[i]

            # Core
            pix_tmp = hp.ang2pix(self.nside,
                                 (self.ps_catalog['Lat (deg)'] + 90) /
                                 180 * np.pi, self.ps_catalog['Lon (deg)'] /
                                 180 * np.pi)
            ps_core = Tb_list[:, 0]
            hpmap[pix_tmp] += ps_core

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
              Area of the PS, e.g., `1.0*au.sr`
        freq: `~astropy.units.Quantity`
              Frequency, e.g., `1.0*au.MHz`

        Return
        -------
        Tb:`~astropy.units.Quantity`
             Average brightness temperature, e.g., `1.0*au.K`
        """
        # Init
        freq_ref = 151 * au.MHz
        freq = freq * au.MHz
        # Luminosity at 151MHz
        lumo_151 = self.lumo.to(au.Jy)  # [W/Hz/Sr to Jy]
        # Calc flux
        # core-to-extend ratio
        ang = self.lobe_ang.to(au.rad).value
        x = np.random.normal(self.xmed, 0.5)
        beta = np.sqrt((self.gamma**2 - 1) / self.gamma)
        B_theta = 0.5 * ((1 - beta * np.cos(ang))**-2 +
                         (1 + beta * np.cos(ang))**-2)
        ratio_obs = 10**x * B_theta
        # Core
        lumo_core = ratio_obs / (1 + ratio_obs) * lumo_151
        a0 = (np.log10(lumo_core.value) - 0.07 *
              np.log10(freq_ref.to(au.GHz).value) +
              0.29 * np.log10(freq_ref.to(au.GHz).value) *
              np.log10(freq_ref.to(au.GHz).value))
        lgs = (a0 + 0.07 * np.log10(freq.to(au.GHz).value) - 0.29 *
               np.log10(freq.to(au.GHz).value) *
               np.log10(freq.to(au.GHz).value))
        flux_core = 10**lgs * au.Jy
        # core area
        npix = hp.nside2npix(self.nside)
        core_area = 4 * np.pi * au.sr / npix
        Tb_core = convert.Fnu_to_Tb(flux_core, core_area, freq)
        # lobe
        lumo_lobe = lumo_151 * (1 - ratio_obs) / (1 + ratio_obs)
        flux_lobe = (freq / freq_ref)**(-0.75) * lumo_lobe
        Tb_lobe = convert.Fnu_to_Tb(flux_lobe, area, freq)

        Tb = [Tb_core.value, Tb_lobe.value]
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
        Tb_list = np.zeros((num_ps, 2))
        # Iteratively calculate Tb
        for i in range(num_ps):
            ps_area = self.ps_catalog['Area (sr)'][i] * au.sr
            Tb_list[i, :] = self.calc_single_Tb(ps_area, freq)

        return Tb_list
