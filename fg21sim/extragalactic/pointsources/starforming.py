# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

# from .psparams import PixelParams
from .base import BasePointSource
from ...utils import grid
from ...utils import convert
from .psparams import PixelParams


class StarForming(BasePointSource):
    """
    Generate star forming point sources, inheritate from PointSource class.

    Reference
    ---------
    [1] Fast cirles drawing
        https://github.com/liweitianux/fg21sim/fg21sim/utils/draw.py
        https://github.com/liweitianux/fg21sim/fg21sim/utils/grid.py
    """

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
        """ Load the configs and set the corresponding class attributes"""
        super()._set_configs()
        # point sources amount
        self.num_ps = self.configs.getn(
            "extragalactic/pointsources/starforming/numps")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/starforming/prefix")
        # redshift bin
        z_type = self.configs.getn(
            "extragalactic/pointsources/starforming/z_type")
        if z_type == 'custom':
            start = self.configs.getn(
                "extragalactic/pointsources/starforming/z_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/starforming/z_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/starforming/z_step")
            self.zbin = np.arange(start, stop + step, step)
        else:
            self.zbin = np.arange(0.1, 10, 0.05)
        # luminosity bin
        lumo_type = self.configs.getn(
            "extragalactic/pointsources/starforming/lumo_type")
        if lumo_type == 'custom':
            start = self.configs.getn(
                "extragalactic/pointsources/starforming/lumo_start")
            stop = self.configs.getn(
                "extragalactic/pointsources/starforming/lumo_stop")
            step = self.configs.getn(
                "extragalactic/pointsources/starforming/lumo_step")
            self.lumobin = np.arange(start, stop + step, step)
        else:
            self.lumobin = np.arange(17, 25.5, 0.1)  # [W/Hz/sr]

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
                rho_mat[:, i] = (rho_l0 * (10**self.lumobin / lumo_star) **
                                 (-alpha) * np.exp(-10**self.lumobin /
                                                   lumo_star) * (1 + z)**k1)
            else:
                rho_mat[:, i] = (rho_l0 * (10**self.lumobin / lumo_star) **
                                 (-alpha) * np.exp(-10**self.lumobin /
                                                   lumo_star) * (1 + z1)**k1)

        return rho_mat

    def get_radius(self):
        """
        Generate the disc diameter of normal starforming galaxies.

        Reference
        ---------
        [1] Wilman et al.,  Eq(7-9),
             "A semi-empirical simulation of the extragalactic radio continuum
             sky for next generation radio telescopes",
             2008, MNRAS, 388, 1335-1348.
             http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W
        """
        # Willman Eq. (8)
        delta = np.random.normal(0, 0.3)
        log_M_HI = 0.44 * np.log10(self.lumo) + 0.48 + delta
        # Willman Eq. (7)
        log_D_HI = ((log_M_HI - (6.52 + np.random.uniform(-0.06, 0.06))) /
                    1.96 + np.random.uniform(-0.04, 0.04))
        # Willman Eq. (9)
        log_D = log_D_HI - 0.23 - np.log10(1 + self.z)
        self.radius = 10**log_D / 2 * 1e-3 * au.Mpc  # [Mpc]
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
        # Radius
        self.radius = self.param.get_angle(self.get_radius())  # [rad]
        # W/Hz/Sr to Jy
        self.lumo = self.lumo / \
            self.dA.to(au.m).value**2 * au.W / au.Hz / au.m / au.m
        self.lumo = self.lumo.to(au.Jy)
        # Area
        self.area = np.pi * self.radius**2  # [sr] ?
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi *
                    180 - 90) * au.deg  # [-90,90]
        self.lon = np.random.uniform(
            0, np.pi * 2) / np.pi * 180 * au.deg  # [0,360]

        ps_list = [self.z, self.dA.value, self.lumo.value, self.lat.value,
                   self.lon.value, self.area.value, self.radius.value]

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
        resolution = 1
        for i in range(num_ps):
            # grid
            ps_radius = self.ps_catalog['radius (rad)'][i] * au.rad
            ps_radius = ps_radius.to(au.deg).value  # radius[rad]
            c_lat = self.ps_catalog['Lat (deg)'][i]  # core_lat [au.deg]
            c_lon = self.ps_catalog['Lon (deg)'][i]  # core_lon [au.deg]
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
