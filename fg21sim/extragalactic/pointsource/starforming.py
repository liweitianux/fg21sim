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

# Defination of starforming
class starforming(base.pointsource):
    """
    Generate star forming point sources, inheritate from PointSource class.
    """
    # Init
    radius = 0 * au.rad
    Lumo_1400 = 0

    def __init__(self, nside = 512,Lumo_1400=1500):
        base.pointsource.__init__(self, nside)
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
        self.Param = psparams.PixelParams(self.z)
        self.dA = self.Param.dA
        self.radius = self.Param.get_angle(self.get_radius()) # [rad]
        # Area
        self.area = np.pi * self.radius**2  #[sr] ?
        # Position
        self.theta = np.random.uniform(0,np.pi)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg

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

