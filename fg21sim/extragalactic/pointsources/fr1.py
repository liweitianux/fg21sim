# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import numpy as np
import healpy as hp
import astropy.units as au

from .psparams import PixelParams
from .base import BasePointSource
from .flux import Flux

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
    """

    def __init__(self,configs):
        super().__init__(configs)
        self.columns.extend(
            ['lobe_maj (rad)', 'lobe_min (rad)', 'lobe_ang (deg)'])
        self.nCols = len(self.columns)
        self._get_configs()

    def _get_configs(self):
        """Load the configs and set the corresponding class attributes"""
        # point sources amount
        self.num_ps = self.configs.getn("extragalactic/pointsources/num_fr1")
        # prefix
        self.prefix = self.configs.getn(
            "extragalactic/pointsources/prefix_fr1")

    def gen_lobe(self):
        D0 = 1 * au.Mpc
        self.lobe_maj = 0.5 * np.random.uniform(
            0, D0.value * (1 + self.z)**(-1.4)) * au.Mpc
        self.lobe_min = self.lobe_maj*np.random.uniform(0.2,1)*au.Mpc
        self.lobe_ang = np.random.uniform(0, np.pi)/ np.pi * 180 * au.deg

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
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA

        # Position
        x = np.random.uniform(0,1)
        self.theta = np.arccos(x)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/ np.pi * 180 * au.deg

        # lobe
        lobe = self.gen_lobe()

        # Area
        self.area = np.pi * self.lobe_maj * self.lobe_min

        ps_list = [self.z, self.dA.value, self.theta.value,
                   self.phi.value, self.area.value]

        ps_list.extend(lobe)

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
        ps_flux = Flux(freq, 4)
        # ps_flux_list
        num_ps = self.ps_catelog.shape[0]
        ps_flux_list = np.zeros((num_ps, 2))
        # Iteratively calculate flux
        for i in range(num_ps):
            ps_area = self.ps_catelog['Area (sr)'][i]
            ps_flux_list[i, :] = ps_flux.calc_Tb(ps_area)[0:2]

        return ps_flux_list

    def draw_single_ps(self, freq):
        """
        Designed to draw the elliptical lobes of FRI and FRII

        Prameters
        ---------
        nside: int and dyadic
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
        num_ps = self.ps_catelog.shape[0]
        # Gen flux list
        ps_flux_list = self.calc_flux(freq)
        ps_lobe = ps_flux_list[:, 1]
        # Iteratively draw ps
        for i in range(num_ps):
            # Parameters
            theta = self.ps_catelog['Theta (deg)'][i] * au.deg
            phi = self.ps_catelog['Phi (deg)'][i] * au.deg
            lobe_maj = self.ps_catelog['lobe_maj (rad)'][i]
            lobe_min = self.ps_catelog['lobe_min (rad)'][i]
            lobe_ang = self.ps_catelog['lobe_ang (deg)'][i] / 180 * np.pi

            # Lobe1
            lobe1_theta = lobe_maj * np.cos(lobe_ang) * au.rad
            lobe1_theta = theta + lobe1_theta.to(au.deg)
            lobe1_phi = lobe_maj * np.sin(lobe_ang) * au.rad
            lobe1_phi = phi + lobe1_phi.to(au.deg)
            # Focuses
            lobe_c = np.sqrt(lobe_maj**2 - lobe_min**2)  # focus distance
            F1_core_x = lobe_c
            F1_core_y = 0
            F2_core_x = -lobe_c
            F2_core_y = 0
            # draw
            step = lobe_maj / 10
            x = np.arange(-lobe_maj, lobe_maj + step, step)
            y = np.arange(-lobe_min, lobe_min + step, step)
            # Ellipse
            for p in range(len(x)):
                for q in range(len(y)):
                    DistFocus1 = np.sqrt(
                        (x[p] - F1_core_x)**2 + (y[q] - F1_core_y)**2)
                    DistFocus2 = np.sqrt(
                        (x[p] - F2_core_x)**2 + (y[q] - F2_core_y)**2)
                    if (DistFocus1 + DistFocus2 <= 2 * lobe_maj):
                        # rotation
                        x_ang = (x[p] * au.rad).to(au.deg)
                        y_ang = (y[q] * au.rad).to(au.deg)
                        x_r = ((x_ang * np.cos(lobe_ang) -
                               y_ang * np.sin(lobe_ang) +
                               lobe1_theta).value / 180 * np.pi)
                        y_r = ((x_ang * np.sin(lobe_ang) +
                               y_ang * np.cos(lobe_ang) +
                               lobe1_phi).value / 180 * np.pi)
                        # Judge and Fill
                        if x_r > np.pi:
                            x_r -= np.pi
                        elif x_r < 0:
                            x_r += np.pi
                        if y_r > 2 * np.pi:
                            y_r -= 2 * np.pi
                        elif y_r < 0:
                            y_r += 2 * np.pi
                        pix_tmp = hp.ang2pix(
                            self.nside, x_r, y_r )
                        hpmap[pix_tmp] += ps_lobe[i]

            # Lobe2
            lobe2_theta = lobe_maj * np.cos(lobe_ang + np.pi) * au.rad
            lobe2_theta = theta + lobe2_theta.to(au.deg)
            lobe2_phi = lobe_maj * np.sin(lobe_ang + np.pi) * au.rad
            lobe2_phi = phi + lobe2_phi.to(au.deg)
            # Focuses
            lobe_c = np.sqrt(lobe_maj**2 - lobe_min**2)  # focus distance
            F1_core_x = lobe_c
            F1_core_y = 0
            F2_core_x = -lobe_c
            F2_core_y = 0
            # draw
            step = lobe_maj / 10
            x = np.arange(-lobe_maj, lobe_maj + step, step)
            y = np.arange(-lobe_min, lobe_min + step, step)
            # Ellipse
            for p in range(len(x)):
                for q in range(len(y)):
                    DistFocus1 = np.sqrt(
                        (x[p] - F1_core_x)**2 + (y[q] - F1_core_y)**2)
                    DistFocus2 = np.sqrt(
                        (x[p] - F2_core_x)**2 + (y[q] - F2_core_y)**2)
                    if (DistFocus1 + DistFocus2 <= 2 * lobe_maj):
                        # rotation
                        x_ang = (x[p] * au.rad).to(au.deg)
                        y_ang = (y[q] * au.rad).to(au.deg)
                        x_r = ((x_ang * np.cos(lobe_ang + np.pi) -
                               y_ang * np.sin(lobe_ang + np.pi) +
                               lobe2_theta).value / 180 * np.pi)
                        y_r = ((x_ang * np.sin(lobe_ang + np.pi) +
                               y_ang * np.cos(lobe_ang + np.pi) +
                               lobe2_phi).value / 180 * np.pi)
                        # Judge and Fill
                        if x_r > np.pi:
                            x_r -= np.pi
                        elif x_r < 0:
                            x_r += np.pi
                        if y_r > 2 * np.pi:
                            y_r -= 2 * np.pi
                        elif y_r < 0:
                            y_r += 2 * np.pi
                        pix_tmp = hp.ang2pix(self.nside, x_r,y_r)
                        hpmap[pix_tmp] += ps_lobe[i]
            # Core
            pix_tmp = hp.ang2pix(self.nside, self.ps_catelog['Theta (deg)'] / 180 *
                                 np.pi, self.ps_catelog['Phi (deg)'] / 180 * np.pi)
            ps_core = ps_flux_list[:, 0]
            hpmap[pix_tmp] += ps_core

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
