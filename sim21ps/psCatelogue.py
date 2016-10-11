#!/usr/bin/env python3
#
# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Simulation of point sources for 21cm signal detection

Point sources types
-----
1. Star forming (SF) galaxies
2. Star bursting (SB) galaxies
3. Radio quiet AGN (RQ_AGN)
4. Faranoff-Riley I (FRI)
5. Faranoff-Riley II (FRII)

References
------------
[1] Wilman R J, Miller L, Jarvis M J, et al.
    A semi-empirical simulation of the extragalactic radio continuum sky for
    next generation radio telescopes[J]. Monthly Notices of the Royal
    Astronomical Society, 2008, 388(3):1335–1348.
[2] Jelic et al.
[3] Healpix, healpy.readthedocs.io
"""
# Import packages or modules
import os
# import healpy as hp  # healpy
import numpy as np
import time
from pandas import DataFrame
import astropy.units as au
# Custom module
import basic_params

# Defination of classes
class PointSource:
    """
    The basic class of point sources

    Parameters
    ----------
    z: float
        Redshift, z ~ U(0,20)
    dA: au.Mpc
        Angular diameter distance, which is calculated according to the cosmology
        constants. In this work, it is calculated by module basic_params
    theta: au.rad
        The colatitude angle in the spherical coordinate system
    phi: au.rad
        The longtitude angle in the spherical coordinate system
    area: au.sr
        Area of the point sources, sr = rad^2

    Functions
    ---------
    gen_sgl_ps
        Generate single ps
    save_as_csv

    """
    # Init
    z = 0
    dA = 0 * au.Mpc * 0
    theta = au.deg * 0
    phi = au.deg * 0
    area = au.sr * 0
    Columns = []
    nCols = 0

    def __init__(self,nside=512):
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.Param = basic_params.PixelParams( self.z)
        self.dA = self.Param.dA
        # Area
        self.area = 4*np.pi/(12*nside**2)
        # PS_list information
        self.Columns = ['z', 'dA (Mpc)', 'Theta (deg)',
                        'Phi (deg)', 'Area (sr)']
        self.nCols = 5

    def gen_sgl_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.Param = basic_params.PixelParams(self.z)
        self.dA = self.Param.dA
        # Position
        self.theta = np.random.uniform(0,np.pi) * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2) * 180 * au.deg

        PS_list = np.array(
            [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value])
        return PS_list

    def save_as_csv(self, NumPS=100, folder_name='PS_tables/'):
        """
        Generae NumPS of point sources and save them into a csv file.
        """
        # Init
        PS_Table = np.zeros((NumPS, self.nCols))
        for x in range(NumPS):
            PS_Table[x, :] = self.gen_sgl_ps()

        # Transform into Dataframe
        PS_frame = DataFrame(PS_Table, columns=self.Columns,
                             index=list(range(NumPS)))

        # Save to csv
        if os.path.exists(folder_name) == False:
            os.mkdir(folder_name)

        file_name = 'PS_' + str(NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'
        PS_frame.to_csv(folder_name + '/' + file_name)
        return PS_frame


class StarForming(PointSource):
    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    # Init
    radius = 0 * au.rad
    Lumo_1400 = 0

    def __init__(self, nside = 512,Lumo_1400=1500):
        PointSource.__init__(self, nside)
        self.Lumo_1400 = Lumo_1400
        self.Columns.append('radius (rad)')
        self.nCols += 1

    def get_radius(self):
        Temp = 0.22 * np.log10(self.Lumo_1400) - np.log10(1 + self.z) - 3.32
        self.radius = 10 ** Temp / 2 * au.Mpc

        return self.radius

    def gen_sgl_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.Param = basic_params.PixelParams(self.z)
        self.dA = self.Param.dA
        self.radius = self.Param.get_angle(self.get_radius()) # [rad]
        # Area
        self.area = np.pi * self.radius**2  #[sr] ?
        # Position
        self.theta = np.random.uniform(0,np.pi) * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2) * 180 * au.deg

        PS_list = np.array(
            [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value, self.radius.value])
        return PS_list

    def save_as_csv(self, NumPS=100, folder_name='PS_tables'):
        """
        Generae NumPS of point sources and save them into a csv file.
        """
        # Init
        PS_Table = np.zeros((NumPS, self.nCols))
        for x in range(NumPS):
            PS_Table[x, :] = self.gen_sgl_ps()

        # Transform into Dataframe
        PS_frame = DataFrame(PS_Table, columns=self.Columns,
                             index=list(range(NumPS)))

        # Save to csv
        if os.path.exists(folder_name) == False:
            os.mkdir(folder_name)

        file_name = 'SF_' + str(NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'
        PS_frame.to_csv(folder_name + '/' + file_name)
        return PS_frame


class StarBursting(PointSource):
    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    # Init
    radius = 0

    def __init__(self,nside=512):
        PointSource.__init__(self,nside)
        self.Columns.append('radius (rad)')
        self.nCols += 1

    def get_radius(self):
        if self.z <= 1.5:
            self.radius = (1 + self.z)**2.5 * 1e-3
        else:
            self.radius = 10 * 1e-3

        return self.radius

    def gen_sgl_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.Param = basic_params.PixelParams(self.z)
        self.dA = self.Param.dA
        self.radius = self.Param.get_angle(self.get_radius())
        # Area
        self.area = np.pi * self.radius**2 #[sr]
        # Position
        self.theta = np.random.uniform(0,np.pi) * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2) * 180 * au.deg

        PS_list = np.array(
            [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value, self.radius.value])
        return PS_list

    def save_as_csv(self, NumPS=100, folder_name='PS_tables'):
        """
        Generae NumPS of point sources and save them into a csv file.
        """
        # Init
        PS_Table = np.zeros((NumPS, self.nCols))
        for x in range(NumPS):
            PS_Table[x, :] = self.gen_sgl_ps()

        # Transform into Dataframe
        PS_frame = DataFrame(PS_Table, columns=self.Columns,
                             index=list(range(NumPS)))

        # Save to csv
        if os.path.exists(folder_name) == False:
            os.mkdir(folder_name)

        file_name = 'SB_' + str(NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'
        PS_frame.to_csv(folder_name + '/' + file_name)

        return PS_frame


class FRI(PointSource):
    """
    Generate Faranoff-Riley I (FRI) AGN

    Parameters
    ----------
    lobe_maj: float
        The major half axis of the lobe
    lobe_min: float
        The minor half axis of the lobe
    lobe_ang: float
        The rotation angle of the lobe from LOS

    """
    # New parameters
    lobe_maj = 0 * au.rad
    lobe_min = 0 * au.rad
    lobe_ang = 0 * au.deg

    def __init__(self,nside=512):
        PointSource.__init__(self, nside)
        self.Columns.extend(
            ['lobe_maj (rad)', 'lobe_min (rad)', 'lobe_ang (deg)'])
        self.nCols += 3

    def gen_lobe(self):
        """
        According to Wang's work, the linear scale at redshift z obeys to U(0,D0(1+z)^(-1.4))
        """
        D0 = 1 * au.Mpc
        self.lobe_maj = 0.5 * np.random.uniform(0, D0.value * (1 + self.z)**(-1.4)) * au.Mpc
        self.lobe_min = self.lobe_maj * np.random.uniform(0.2, 1) * au.Mpc
        self.lobe_ang = np.random.uniform(0, np.pi) * 180 * au.deg

        # Transform to pixel
        self.lobe_maj = self.Param.get_angle(self.lobe_maj)
        self.lobe_min = self.Param.get_angle(self.lobe_min)
        lobe = [self.lobe_maj.value, self.lobe_min.value, self.lobe_ang.value]

        return lobe

    def gen_sgl_ps(self):
        """
        Generate single point source, and return its data as a list.

        """
        # Redshift
        self.z = np.random.uniform(0, 20)
        # angular diameter distance
        self.Param = basic_params.PixelParams(self.z)
        self.dA = self.Param.dA

        # Position
        self.theta = np.random.uniform(0,np.pi) * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2) * 180 * au.deg

        # Area
        self.area = np.pi * self.lobe_maj * self.lobe_min

        # lobe
        lobe = self.gen_lobe()

        PS_list = [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value]
        PS_list.extend(lobe)

        PS_list = np.array(PS_list)
        return PS_list

    def save_as_csv(self, NumPS=100, folder_name='PS_tables'):
        """
        Generae NumPS of point sources and save them into a csv file.
        """
        # Init
        PS_Table = np.zeros((NumPS, self.nCols))
        for x in range(NumPS):
            PS_Table[x, :] = self.gen_sgl_ps()

        # Transform into Dataframe
        PS_frame = DataFrame(PS_Table, columns=self.Columns,
                             index=list(range(NumPS)))

        # Save to csv
        if os.path.exists(folder_name) == False:
            os.mkdir(folder_name)

        file_name = 'FRI_' + str(NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'
        PS_frame.to_csv(folder_name + '/' + file_name)

        return PS_frame


class FRII(FRI):
    """
    Generate Faranoff-Riley I (FRI) AGN, a class inherit from FRI
    """

    def __init__(self,nside = 512):
        FRI.__init__(self,nside)

    def gen_lobe(self):
        """
        According to Wang's work, the linear scale at redshift z obeys to U(0,D0(1+z)^(-1.4))
        """
        D0 = 1
        self.lobe_maj = 0.5 * np.random.uniform(0, D0 * (1 + self.z)**(-1.4)) * au.Mpc
        self.lobe_min = self.lobe_maj * np.random.uniform(0.2, 1) * au.Mpc
        self.lobe_ang = np.random.uniform(0, np.pi / 3) * 180 * au.deg # Different from FRI

        # Transform to pixel
        self.lobe_maj = self.Param.get_angle(self.lobe_maj)
        self.lobe_min = self.Param.get_angle(self.lobe_min)
        lobe = [self.lobe_maj.value, self.lobe_min.value, self.lobe_ang.value]

        return lobe
