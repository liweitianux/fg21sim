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
import psparams

# Defination of the base class
class pointsource:
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
    #Init
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
        self.Param = psparams.PixelParams( self.z)
        self.dA = self.Param.dA
        # Area
        self.nside = nside
        self.area = 4*np.pi/(12*nside**2) * au.sr
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
        self.Param = psparams.PixelParams(self.z)
        self.dA = self.Param.dA
        # Position
        self.theta = np.random.uniform(0,np.pi)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg
        # Area
        self.area = 4*np.pi/(12*self.nside**2) * au.sr        

        PS_list = np.array(
            [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value])
        return PS_list

    def save_as_csv(self, NumPS=100, folder_name='PS_tables/'):
        """
        Generate NumPS of point sources and save them into a csv file.
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
