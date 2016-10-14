# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

import os

import numpy as np
import pandas as pd
import astropy.units as au
import healpy as hp

from .flux import Flux
from .starforming import StarForming
from .starbursting import StarBursting
from .radioquiet import RadioQuiet
from .fr1 import FRI
from .fr2 import FRII


class PointSources:
    """
    This class namely pointsource is designed to generate PS catelogues,
    read csv format PS lists, calculate the flux and surface brightness
    of the sources at different frequencies, and then ouput hpmaps

    functions
    ---------
    read_csv
        read the csv format files, judge the PS type and
        transformed to be iterable numpy.ndarray.

    calc_flux
        calculate the flux and surface brightness of the PS.

    draw_elp
        processing on the elliptical and circular core or lobes.

    draw_circle
        processing on the circular star forming or bursting galaxies

    draw_ps
        generate hpmap with respect the imput PS catelogue
"""

    def __init__(self, configs):
        self.configs = configs
        self._get_configs()
        self.files = []

    def _get_configs(self):
        """Load configs and set the attributes"""
        # nside of the healpix cell
        self.nside = self.configs.getn("common/nside")
        # frequencies
        self.freq = self.configs.getn("frequency/frequencies")
        # save flag
        self.save = self.configs.getn("extragalactic/pointsource/save")

    def gen_catelogue(self):
        """Generate the catelogues"""
        # Init
        sf = StarForming(self.configs)
        sb = StarBursting(self.configs)
        rq = RadioQuiet(self.configs)
        fr1 = FRI(self.configs)
        fr2 = FRII(self.configs)

        # Save
        if self.save:
            self.files.append(sf.save_as_csv()[1])
            self.files.append(sb.save_as_csv()[1])
            self.files.append(rq.save_as_csv()[1])
            self.files.append(fr1.save_as_csv()[1])
            self.files.append(fr2.save_as_csv()[1])

    def read_csv(self,filepath):
        """
        Read csv format point source files,judge its class
        type according to its name.

        Parameters
        ----------
        filepath: str
            Path of the file.
        """
        # Split to folder name and file name
        filename = os.path.basename(filepath)
        # Split and judge point source type
        class_list = ['SF', 'SB', 'RQ', 'FRI', 'FRII']
        class_name = filename.split('.')[0]
        ps_type = class_list.index(class_name) + 1
        # Read csv
        ps_data = pd.read_csv(filepath)

        return ps_type, ps_data


    def calc_flux(self, ps_type, freq, ps_data):
        """
        Calculate the flux and surface brightness of the point source.

        Parameters
        ----------
        ps_type: int
            Type of point source
        freq: float
            frequency
        ps_data: pandas.core.frame.DataFrame
            Data of the point sources
        """
        # init flux
        ps_flux = Flux(freq, ps_type)
        # ps_flux_list
        num_ps = ps_data.shape[0]
        if ps_type <= 3:
            ps_flux_list = np.zeros((num_ps,))
            # Iteratively calculate flux
            for i in range(num_ps):
                ps_area = ps_data['Area (sr)'][i]
                ps_flux_list[i] = ps_flux.calc_Tb(ps_area)
        else:
            ps_flux_list = np.zeros((num_ps, 2))
            # Iteratively calculate flux
            for i in range(num_ps):
                ps_area = ps_data['Area (sr)'][i]
                ps_flux_list[i, :] = ps_flux.calc_Tb(ps_area)[0:2]

        return ps_flux_list

    def draw_rq(self, ps_data, freq):
        """
        Designed to draw the radio quiet AGN

        Parameters
        ----------
        ImgMat: np.ndarray
            Two dimensional matrix, to describe the image
        ps_data: pandas.core.frame.DataFrame
            Data of the point sources
        freq: float
            frequency
        """
        # Init
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        # Gen flux list
        ps_flux_list = self.calc_flux(3, freq, ps_data)
        # Angle to pix
        theta = ps_data['Theta (deg)'] / 180 * np.pi
        phi = ps_data['Phi (deg)'] / 180 * np.pi
        pix = hp.ang2pix(self.nside, theta, phi)
        # Gen hpmap
        hpmap[pix] += ps_flux_list

        return hpmap

    def draw_circle(self, ps_data, ps_type, freq):
        """
        Designed to draw the circular  star forming  and star bursting ps.

        Prameters
        ---------
        nside: int and dyadic
            number of sub pixel in a cell of the healpix structure
        ps_data: pandas.core.frame.DataFrame
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
        ps_flux_list = self.calc_flux(ps_type, freq, ps_data)
        #  Iteratively draw the ps
        num_ps = ps_data.shape[0]
        for i in range(num_ps):
            # grid
            ps_radius = ps_data['radius (rad)'][i]  # radius[rad]
            theta = ps_data['Theta (deg)'][i] * au.deg   # theta
            phi = ps_data['Phi (deg)'][i] * au.deg  # phi
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

    def draw_lobe(self, ps_data, ps_type, freq):
        """
        Designed to draw the elliptical lobes of FRI and FRII

        Prameters
        ---------
        nside: int and dyadic
        ps_data: pandas.core.frame.DataFrame
            Data of the point sources
        ps_type: int
            Class type of the point soruces
        freq: float
            frequency
        """
        # Init
        npix = hp.nside2npix(self.nside)
        hpmap = np.zeros((npix,))
        num_ps = ps_data.shape[0]
        # Gen flux list
        ps_flux_list = self.calc_flux(ps_type, freq, ps_data)
        ps_lobe = ps_flux_list[:, 1]
        # Iteratively draw ps
        for i in range(num_ps):
            # Parameters
            theta = ps_data['Theta (deg)'][i] * au.deg
            phi = ps_data['Phi (deg)'][i] * au.deg
            lobe_maj = ps_data['lobe_maj (rad)'][i]
            lobe_min = ps_data['lobe_min (rad)'][i]
            lobe_ang = ps_data['lobe_ang (deg)'][i] / 180 * np.pi

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
            pix_tmp = hp.ang2pix(self.nside, ps_data['Theta (deg)'] / 180 *
                                 np.pi, ps_data['Phi (deg)'] / 180 * np.pi)
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

        # load csv
        for filepath in self.files:
            ps_type, ps_data = self.read_csv(filepath)

            # get hpmaps
            if ps_type == 1 or ps_type == 2:
                for i in range(num_freq):
                    hpmaps[:, i] = self.draw_circle(ps_data, ps_type,
                                             self.freq[i])
            elif ps_type == 3:
                for i in range(num_freq):
                    hpmaps[:, i] = self.draw_rq(ps_data, self.freq[i])
            else:
                for i in range(num_freq):
                    hpmaps[:, i] = self.draw_lobe(ps_data, ps_type,
                                           self.freq[i])

        return hpmaps
