# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource
from .flux import Flux
from fg21sim.utils import grid

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
        self.num_ps = self.configs.getn("extragalactic/pointsources/num_sb")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/prefix_sb")

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
        self.lat = np.arccos(x)/np.pi * 180 * au.deg
        self.lon = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

        ps_list = [self.z, self.dA.value, self.lat.value,
                   self.lon.value, self.area.value, self.radius.value]

        return ps_list

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
        ps_flux = Flux(freq, 2)
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
        Designed to draw the circular  star forming  and star bursting ps.

        Prameters
        ---------
        nside: int and dyadic
            number of sub pixel in a cell of the healpix structure
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
        #  Iteratively draw the ps
        num_ps = self.ps_catelog.shape[0]
        resolution = 1
        for i in range(num_ps):
            # grid
            ps_radius = self.ps_catelog['radius (rad)'][i] * au.rad  
            ps_radius = ps_radius.to(au.deg).value # radius[rad]
            c_lat = self.ps_catelog['Lat (deg)'][i] # core_lat [au.deg]
            c_lon = self.ps_catelog['Lon (deg)'][i] # core_lon [au.deg]
            # Fill with circle
            lon,lat,gridmap = grid.make_grid_ellipse(
                (c_lon,c_lat),(2*ps_radius,2*ps_radius),resolution)
            indices,values = grid.map_grid_to_healpix(
                (lon,lat,gridmap),self.nside)
            hpmap[indices] += ps_flux_list[i]

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
