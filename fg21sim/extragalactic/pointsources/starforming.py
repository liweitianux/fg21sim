# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource
from ...utils import grid
from ...utils import convert


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

    def _set_configs(self):
        """ Load the configs and set the corresponding class attributes"""
        super()._set_configs()
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
        self.radius = self.param.get_angle(self.get_radius())  # [rad]
        # Area
        self.area = np.pi * self.radius**2  # [sr] ?
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi *
                    180 - 90) * au.deg  # [-90,90]
        self.lon = np.random.uniform(
            0, np.pi * 2) / np.pi * 180 * au.deg  # [0,360]

        ps_list = [self.z, self.dA.value, self.lat.value,
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
