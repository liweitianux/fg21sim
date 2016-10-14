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
[1] Wilman et al.,
    "A semi-empirical simulation of the extragalactic radio continuum
    sky for next generation radio telescopes",
    2008, MNRAS, 388, 1335-1348.
    http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W
[2] Jelic et al.,
    "Foreground simulations for the LOFAR-Epoch of Reionization
    Experiment",
    2008, MNRAS, 389, 1319-1335.
    http://adsabs.harvard.edu/abs/2008MNRAS.389.1319W
[3] Spherical uniform distribution
    https://www.jasondavies.com/maps/random-points/
"""
import os

import numpy as np
import pandas as pd

import healpy as hp
import astropy.units as au

from .psparams import PixelParams

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
        # save flag
        self.save = self.configs.getn("extragalactic/pointsources/save")
        # Output_dir
        self.output_dir = self.configs.getn(
                            "extragalactic/pointsources/output_dir")


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
        self.phi = np.random.uniform(0,np.pi*2)/np.pi * 180 * au.deg
        # Area
        npix = hp.nside2npix(self.nside)
        self.area = 4*np.pi/npix * au.sr

        ps_list = [self.z, self.dA.value, self.theta.value,
                   self.phi.value, self.area.value]

        return ps_list

    def save_as_csv(self):
        """
        Generate num_ps of point sources and save them into a csv file.
        """
        # Init
        ps_table = np.zeros((self.num_ps, self.nCols))
        for x in range(self.num_ps):
            ps_table[x, :] = self.gen_single_ps()

        # Transform into Dataframe
        ps_frame = pd.DataFrame(ps_table, columns=self.columns,
                             index=list(range(self.num_ps)))

        # Save to csv
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        pattern = "{prefix}.csv"
        filename = pattern.format(prefix = self.prefix)

        # save to csv
        if self.save:
            file_name = os.path.join(self.output_dir, filename)
            ps_frame.to_csv(filename)

        return ps_frame, file_name
