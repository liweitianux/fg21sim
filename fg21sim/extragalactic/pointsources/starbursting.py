# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource
from .flux import Flux


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
        self.theta = np.arccos(x)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

        ps_list = [self.z, self.dA.value, self.theta.value,
                   self.phi.value, self.area.value, self.radius.value]

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
        ps_type: int
            Class type of the point soruces
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
        for i in range(num_ps):
            # grid
            ps_radius = self.ps_catelog['radius (rad)'][i]  # radius[rad]
            theta = self.ps_catelog['Theta (deg)'][i] * au.deg   # theta
            phi = self.ps_catelog['Phi (deg)'][i] * au.deg  # phi
            # Fill with circle
            step = ps_radius / 10  # Should be fixed
            # x and y are the differencial rad to the core point at the theta and
            # phi directions.
            x = np.arange(-ps_radius, ps_radius + step, step) * au.rad
            y = np.arange(- ps_radius,  ps_radius + step, step) * au.rad
            for p in range(len(x)):
                for q in range(len(y)):
                    if np.sqrt(x[p].value**2 + y[q].value**2) <= ps_radius:
                        x_ang = (x[p].to(au.deg) + theta).value / 180 * np.pi
                        y_ang = (y[q].to(au.deg) + phi).value / 180 * np.pi
                        if x_ang > np.pi:
                            x_ang -= np.pi
                        elif x_ang < 0:
                            x_ang += np.pi
                        if y_ang > 2 * np.pi:
                            y_ang -= 2 * np.pi
                        elif y_ang < 0:
                            y_ang += 2 * np.pi
                        pix_tmp = hp.ang2pix(
                            self.nside, x_ang, y_ang)
                        hpmap[pix_tmp] += ps_flux_list[i]

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
