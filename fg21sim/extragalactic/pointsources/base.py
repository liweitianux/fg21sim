# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Simulation of point sources for 21cm signal detection

Point sources types
-------------------
1. Star forming (SF) galaxies
2. Star bursting (SB) galaxies
3. Radio quiet AGN (RQ_AGN)
4. Faranoff-Riley I (FRI)
5. Faranoff-Riley II (FRII)

References
----------
[1] Wilman R J, Miller L, Jarvis M J, et al.
    A semi-empirical simulation of the extragalactic radio continuum sky for
    next generation radio telescopes[J]. Monthly Notices of the Royal
    Astronomical Society, 2008, 388(3):1335–1348.
[2] Jelic, V., Zaroubi S., Labropoulos. P, et al. Foreground simulations for
    the LOFAR-Epoch of Reionization Experiment [J]. Monthly Notices of the Royal
    Astronomical Society, 2008, 389(3):1319-1335.
"""

import os
import numpy as np
import healpy as hp
import time
from pandas import DataFrame
import astropy.units as au
from .psparams import PixelParams


# Defination of the base class
class BasePointSource:
    """
    The basic class of point sources

    Parameters
    ----------
    z: float;
        Redshift, z ~ U(0,20)
    dA: au.Mpc;
        Angular diameter distance, which is calculated according to the cosmology
        constants. In this work, it is calculated by module basic_params
    theta: au.rad;
        The colatitude angle in the spherical coordinate system
    phi: au.rad;
        The longtitude angle in the spherical coordinate system
    area: au.sr;
        Area of the point sources, sr = rad^2

    """
    #Init
    def __init__(self,configs):
        # configures
        self.configs = configs
        # PS_list information
        self.columns = ['z', 'dA (Mpc)', 'Theta (deg)',
                        'Phi (deg)', 'Area (sr)']
        self.nCols = len(self.columns)
        self._get_base_configs()

    def _get_base_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        # common
        self.nside = self.configs.getn("common/nside")
        # prefix
        # self.prefix = self.configs.getn("extragalctic/pointsource/prefix")
        # save flag
        self.save = self.configs.getn("extragalactic/pointsource/save")
        # Output_dir
        self.foldname = self.configs.getn(
        "extragalactic/pointsource/output_dir")
        # Number of point sources
        # NumPS = self.configs.getn("extragalactic/pointsource/Num_PS")

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
        self.theta = np.random.uniform(0,np.pi)/np.pi * 180 * au.deg
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg
        # Area
        npix = hp.nside2npix(self.nside)
        self.area = 4*np.pi/npix * au.sr

        ps_list = np.array(
            [self.z, self.dA.value, self.theta.value, self.phi.value, self.area.value])
        return ps_list

    def save_as_csv(self):
        """
        Generate NumPS of point sources and save them into a csv file.
        """
        # Init
        ps_table = np.zeros((self.NumPS, self.nCols))
        for x in range(self.NumPS):
            ps_table[x, :] = self.gen_single_ps()

        # Transform into Dataframe
        ps_frame = DataFrame(ps_table, columns=self.columns,
                             index=list(range(self.NumPS)))

        # Save to csv
        if os.path.exists(self.foldname) == False:
            os.mkdir(self.foldname)

        file_name = self.prefix +'_'+str(self.NumPS) + '_' + \
            time.strftime('%Y%m%d_%H%M%S') + '.csv'

        # save to csv
        if self.save:
            file_name = self.foldname + '/' + file_name
            ps_frame.to_csv(file_name)

        return ps_frame,file_name
