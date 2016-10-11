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
import base

# Defination of Faranoff-RileyI AGN
class fr1(base.pointsource):
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
        base.pointsource.__init__(self, nside)
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
        self.lobe_ang = np.random.uniform(0, np.pi)/ np.pi * 180 * au.deg

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
        self.Param = psparams.PixelParams(self.z)
        self.dA = self.Param.dA

        # Position
        self.theta = np.random.uniform(0,np.pi)/ np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/ np.pi * 180 * au.deg

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
