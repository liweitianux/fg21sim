# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp

from .base import BasePointSource
from .psparams import PixelParams
from ...utils import grid
from ...utils import convert


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
        The rotation angle of the lobe correspoind to line of sight

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
        self.xmed = -2.8
        # Lorentz factor of the jet
        self.gamma = 8
        # Number density matrix
        self.rho_mat = self.calc_number_density()
        # Cumulative distribution of z and lumo
        self.cdf_z, self.cdf_lumo = self.calc_cdf()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        pscomp = "extragalactic/pointsources/FRII/"
        # point sources amount
        self.num_ps = self.configs.getn(pscomp+"numps")
        # prefix
        self.prefix = self.configs.getn(pscomp+"prefix")
        # redshift bin
        z_type = self.configs.getn(pscomp+"z_type")
        if z_type == 'custom':
            start = self.configs.getn(pscomp+"z_start")
            stop = self.configs.getn(pscomp+"z_stop")
            step = self.configs.getn(pscomp+"z_step")
            self.zbin = np.arange(start, stop + step, step)
        else:
            self.zbin = np.arange(0.1, 10, 0.05)
        # luminosity bin
        lumo_type = self.configs.getn(pscomp+"lumo_type")
        if lumo_type == 'custom':
            start = self.configs.getn(pscomp+"lumo_start")
            stop = self.configs.getn(pscomp+"lumo_stop")
            step = self.configs.getn(pscomp+"lumo_step")
            self.lumobin = np.arange(start, stop + step, step)
        else:
            self.lumobin = np.arange(25.5, 30.5, 0.1)  # [W/Hz/sr]

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
        alpha = 2.27  # spectral index
        lumo_star = 10.0**26.95  # critical luminosity
        rho_l0 = 10.0**(-6.196)  # normalization constant
        z0 = 1.91  # center redshift
        z2 = 1.378  # variance
        # Calculation
        for i, z in enumerate(self.zbin):
            # space density revolusion
            fh = np.exp(-0.5 * (z - z0)**2 / z2**2)
            rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                              (-alpha) *
                              np.exp(-lumo_star / 10.0**self.lumobin)) *
                             fh)

        return rho_mat

    def gen_lobe(self):
        """
        Calculate lobe parameters

        References
        ----------
        [1] Wilman et al.,
             "A semi-empirical simulation of the extragalactic radio continuum
             sky for next generation radio telescopes",
             2008, MNRAS, 388, 1335-1348.
             http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W

        Return
        ------
        lobe: list
            lobe = [lobe_maj, lobe_min, lobe_ang], which represent the major
            and minor axes and the rotation angle.
        """
        D0 = 1  # [Mpc]
        self.lobe_maj = 0.5 * np.random.uniform(0, D0 * (1 + self.z)**(-1.4))
        self.lobe_min = self.lobe_maj * np.random.uniform(0.2, 1)
        self.lobe_ang = np.random.uniform(0, np.pi) / np.pi * 180

        # Transform to pixel
        self.lobe_maj = self.param.get_angle(self.lobe_maj)
        self.lobe_min = self.param.get_angle(self.lobe_min)
        lobe = [self.lobe_maj, self.lobe_min, self.lobe_ang]

        return lobe

    def gen_single_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift and luminosity
        self.z, self.lumo = self.get_lumo_redshift()
        self.lumo_sr = self.lumo
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA
        # W/Hz/Sr to Jy
        dA = self.dA * 3.0856775814671917E+22  # Mpc to meter
        self.lumo = self.lumo / dA**2 / (10.0**-24)  # [Jy]
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi * 180 - 90)  # [deg]
        self.lon = np.random.uniform(0, np.pi * 2) / np.pi * 180  # [deg]
        # lobe
        lobe = self.gen_lobe()
        # Area
        self.area = np.pi * self.lobe_maj * self.lobe_min

        ps_list = [self.z, self.dA, self.lumo, self.lat, self.lon, self.area]
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
        resolution = self.resolution / 60  # [degree]
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        num_ps = self.ps_catalog.shape[0]
        # Gen flux list
        Tb_list = self.calc_Tb(freq)
        ps_lobe = Tb_list[:, 1]
        # Iteratively draw ps
        for i in range(num_ps):
            # Parameters
            c_lat = self.ps_catalog['Lat (deg)'][i]  # core lat [deg]
            c_lon = self.ps_catalog['Lon (deg)'][i]  # core lon [au.deg]
            lobe_maj = self.ps_catalog['lobe_maj (rad)'][
                i] * 180 / np.pi  # [deg]
            lobe_min = self.ps_catalog['lobe_min (rad)'][
                i] * 180 / np.pi  # [deg]
            lobe_ang = self.ps_catalog['lobe_ang (deg)'][
                i] / 180 * np.pi  # [rad]
            # Offset to the core, refer to Willman Sec2.5.vii
            offset = lobe_maj * 2 * np.random.uniform(0.2, 0.8)
            # Lobe1
            lobe1_lat = (lobe_maj / 2 + offset) * np.cos(lobe_ang)
            lobe1_lat = c_lat + lobe1_lat
            lobe1_lon = (lobe_maj / 2 + offset) * np.sin(lobe_ang)
            lobe1_lon = c_lon + lobe1_lon
            # draw
            # Fill with ellipse
            lon, lat, gridmap = grid.make_grid_ellipse(
                (lobe1_lon, lobe1_lat), (lobe_maj, lobe_min),
                resolution, lobe_ang / np.pi * 180)
            indices, values = grid.map_grid_to_healpix(
                (lon, lat, gridmap), self.nside)
            hpmap[indices] += ps_lobe[i]

            # lobe1_hotspot
            lobe1_hot_lat = (lobe_maj + offset) * np.cos(lobe_ang)
            lobe1_hot_lat = (c_lat + 90 + lobe1_lat) / 180 * np.pi
            lobe1_hot_lon = (lobe_maj + offset) * np.sin(lobe_ang)
            lobe1_hot_lon = (c_lon + lobe1_lon) / 180 * np.pi
            if lobe1_hot_lat < 0:
                lobe1_hot_lat += np.pi
            elif lobe1_hot_lat > np.pi:
                lobe1_hot_lat -= np.pi
            lobe1_hot_index = hp.ang2pix(
                self.nside, lobe1_hot_lat, lobe1_hot_lon)
            hpmap[lobe1_hot_index] += Tb_list[i, 2]

            # Lobe2
            lobe2_lat = (lobe_maj / 2) * np.cos(lobe_ang + np.pi)
            lobe2_lat = c_lat + lobe2_lat
            lobe2_lon = (lobe_maj / 2) * np.sin(lobe_ang + np.pi)
            lobe2_lon = c_lon + lobe2_lon
            # draw
            # Fill with ellipse
            lon, lat, gridmap = grid.make_grid_ellipse(
                (lobe2_lon, lobe2_lat), (lobe_maj, lobe_min),
                resolution, lobe_ang / np.pi * 180)
            indices, values = grid.map_grid_to_healpix(
                (lon, lat, gridmap), self.nside)
            hpmap[indices] += ps_lobe[i]

            # lobe2_hotspot
            lobe2_hot_lat = (lobe_maj + offset) * np.cos(lobe_ang + np.pi)
            lobe2_hot_lat = (c_lat + 90 + lobe1_lat) / 180 * np.pi
            lobe2_hot_lon = (lobe_maj + offset) * np.sin(lobe_ang + np.pi)
            lobe2_hot_lon = (c_lon + lobe1_lon) / 180 * np.pi
            if lobe2_hot_lat < 0:
                lobe2_hot_lat += np.pi
            elif lobe2_hot_lat > np.pi:
                lobe2_hot_lat -= np.pi
            lobe2_hot_index = hp.ang2pix(
                self.nside, lobe2_hot_lat, lobe2_hot_lon)
            hpmap[lobe2_hot_index] += Tb_list[i, 2]

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
        ------
        Tb:`~astropy.units.Quantity`
             Average brightness temperature, e.g., `1.0*au.K`
        """
        # Init
        freq_ref = 151  # [MHz]
        freq = freq  # [MHz]
        # Luminosity at 151MHz
        lumo_151 = self.lumo  # [Jy]
        # Calc flux
        # core-to-extend ratio
        ang = self.lobe_ang / 180 * np.pi
        x = np.random.normal(self.xmed, 0.5)
        beta = np.sqrt((self.gamma**2 - 1) / self.gamma)
        B_theta = 0.5 * ((1 - beta * np.cos(ang))**-2 +
                         (1 + beta * np.cos(ang))**-2)
        ratio_obs = 10**x * B_theta
        # Core
        lumo_core = ratio_obs / (1 + ratio_obs) * lumo_151
        a0 = (np.log10(lumo_core) - 0.07 *
              np.log10(freq_ref * 10.0E-3) +
              0.29 * np.log10(freq_ref * 10.0E-3) *
              np.log10(freq_ref * 10.0E-3))
        lgs = (a0 + 0.07 * np.log10(freq * 10.0E-3) - 0.29 *
               np.log10(freq * 10.0E-3) *
               np.log10(freq * 10.0E-3))
        flux_core = 10**lgs  # [Jy]
        # core area
        npix = hp.nside2npix(self.nside)
        core_area = 4 * np.pi / npix  # [sr]
        Tb_core = convert.Fnu_to_Tb_fast(flux_core, core_area, freq)  # [K]
        # lobe
        lumo_lobe = lumo_151 * (1 - ratio_obs) / (1 + ratio_obs)  # [Jy]
        flux_lobe = (freq / freq_ref)**(-0.75) * lumo_lobe
        Tb_lobe = convert.Fnu_to_Tb_fast(flux_lobe, area, freq)  # [K]

        # hotspots
        # Willman Eq. (3)
        f_hs = (0.4 * (np.log10(self.lumo_sr) - 25.5) +
                np.random.uniform(-0.5, 0.5))
        Tb_hotspot = Tb_lobe * (1 + f_hs)
        Tb = [Tb_core, Tb_lobe, Tb_hotspot]

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
        Tb_list = np.zeros((num_ps, 3))
        # Iteratively calculate Tb
        for i in range(num_ps):
            ps_area = self.ps_catalog['Area (sr)'][i]  # [sr]
            Tb_list[i, :] = self.calc_single_Tb(ps_area, freq)

        return Tb_list
