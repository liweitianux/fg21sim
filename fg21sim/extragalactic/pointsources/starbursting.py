# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp

from .psparams import PixelParams
from .base import BasePointSource
from ...utils import grid
from ...utils import convert


class StarBursting(BasePointSource):

    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    # Init

    def __init__(self, configs):
        super().__init__(configs)
        self.columns.append('radius (rad)')
        self.nCols = len(self.columns)
        self._set_configs()
        # Number density matrix
        self.rho_mat = self.calc_number_density()
        # Cumulative distribution of z and lumo
        self.cdf_z, self.cdf_lumo = self.calc_cdf()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        pscomp = "extragalactic/pointsources/starbursting/"
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
            self.lumobin = np.arange(21, 27, 0.1)  # [W/Hz/sr]

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

        Returns
        -------
        rho_mat: np.ndarray
            Number density matris (joint-distribution of luminosity and
            reshift).
        """
        # Init
        rho_mat = np.zeros((len(self.lumobin), len(self.zbin)))
        # Parameters
        # Refer to Willman's section 2.4
        alpha = 0.7  # spectral index
        lumo_star = 10.0**22  # critical luminosity at 1400MHz
        rho_l0 = 10.0**(-7)  # normalization constant
        z1 = 1.5  # cut-off redshift
        k1 = 3.1  # index of space density revolution
        # Calculation
        for i, z in enumerate(self.zbin):
            if z <= z1:
                rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                                  (-alpha) *
                                  np.exp(-10**self.lumobin / lumo_star)) *
                                 (1 + z)**k1)
            else:
                rho_mat[:, i] = ((rho_l0 * (10**self.lumobin / lumo_star) **
                                  (-alpha) *
                                  np.exp(-10**self.lumobin / lumo_star)) *
                                 (1 + z1)**k1)
        return rho_mat

    def get_radius(self):
        if self.z <= 1.5:
            self.radius = (1 + self.z)**2.5 * 1e-3 / 2  # [Mpc]
        else:
            self.radius = 10 * 1e-3 / 2  # [Mpc]

        return self.radius

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
        dA = self.dA * 3.0856775814671917E+22  # Mpc to meter
        self.lumo = self.lumo / dA**2 / (10.0**-24)  # [Jy]
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi * 180 - 90)  # [deg]
        self.lon = np.random.uniform(0, np.pi * 2) / np.pi * 180  # [deg]
        # Radius
        self.radius = self.param.get_angle(self.get_radius())  # [rad]
        # Area
        self.area = np.pi * self.radius**2  # [sr]

        ps_list = [self.z, self.dA, self.lumo, self.lat, self.lon,
                   self.area, self.radius]

        return ps_list

    def draw_single_ps(self, freq):
        """
        Designed to draw the circular  star forming  and star bursting ps.

        Prameters
        ---------
        nside: int and dyadic
            number of sub pixel in a cell of the healpix structure
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
        resolution = self.resolution / 60  # [degree]
        for i in range(num_ps):
            # grid
            ps_radius = self.ps_catalog['radius (rad)'][i]  # [rad]
            ps_radius = ps_radius * 180 / np.pi  # radius[rad]
            c_lat = self.ps_catalog['Lat (deg)'][i]  # core_lat [deg]
            c_lon = self.ps_catalog['Lon (deg)'][i]  # core_lon [deg]
            # Fill with circle
            lon, lat, gridmap = grid.make_grid_ellipse(
                (c_lon, c_lat), (2 * ps_radius, 2 * ps_radius), resolution)
            indices, values = grid.map_grid_to_healpix(
                (lon, lat, gridmap), self.nside)
            hpmap[indices] += Tb_list[i]

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
        area: float
            Area of the PS
            Unit: [arcsec^2]
        freq: `~astropy.units.Quantity`
              Frequency, e.g., `1.0*au.MHz`

        Return
        ------
        Tb:`~astropy.units.Quantity`
             Average brightness temperature, e.g., `1.0*au.K`
        """
        # Init
        freq_ref = 1400  # [MHz]
        freq = freq  # [MHz]
        # Luminosity at 1400MHz
        lumo_1400 = self.lumo  # [Jy]
        # Calc flux
        flux = (freq / freq_ref)**(-0.7) * lumo_1400
        # Calc brightness temperature
        Tb = convert.Fnu_to_Tb_fast(flux, area, freq)

        return Tb

    def calc_Tb(self, freq):
        """
        Calculate the surface brightness  temperature of the point sources.

        Parameters
        ------------
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
        sr_to_arcsec2 = (np.rad2deg(1) * 3600) ** 2  # [sr] -> [arcsec^2]
        # Iteratively calculate Tb
        for i in range(num_ps):
            ps_area = self.ps_catalog['Area (sr)'][i]  # [sr]
            area = ps_area * sr_to_arcsec2
            Tb_list[i] = self.calc_single_Tb(area, freq)

        return Tb_list
