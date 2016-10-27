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

from .psparams import PixelParams


class BasePointSource:
    """
    The basic class of point sources

    Parameters
    ----------
    z: float;
        Redshift, z ~ U(0,20)
    dA: au.Mpc;
        Angular diameter distance, which is calculated according to the
        cosmology constants. In this work, it is calculated by module
        basic_params
        lumo: au.Jy;
            Luminosity at the reference frequency.
    lat: au.deg;
        The colatitude angle in the spherical coordinate system
    lon: au.deg;
        The longtitude angle in the spherical coordinate system
    area: au.sr;
        Area of the point sources, sr = rad^2

    """
    # Init

    def __init__(self, configs):
        # configures
        self.configs = configs
        # PS_list information
        self.columns = ['z', 'dA (Mpc)', 'luminosity (Jy)', 'Lat (deg)',
                        'Lon (deg)', 'Area (sr)']
        self.nCols = len(self.columns)
        self._set_configs()

    def _set_configs(self):
        """
        Load the configs and set the corresponding class attributes.
        """
        comp = "extragalactic/pointsources/"
        # common
        self.nside = self.configs.getn("common/nside")
        # resolution
        self.resolution = self.configs.getn(comp+"resolution")
        # save flag
        self.save = self.configs.getn(comp+"save")
        # Output_dir
        self.output_dir = self.configs.get_path(comp+"output_dir")

    def calc_number_density(self):
        pass

    def calc_cdf(self):
        """
        Calculate cumulative distribution functions for simulating of
        samples with corresponding reshift and luminosity.

        Parameter
        -----------
        rho_mat: np.ndarray rho(lumo,z)
            The number density matrix (joint-distribution of z and flux)
            of this type of PS.

        Returns
        -------
        cdf_z, cdf_lumo: np.ndarray
            Cumulative distribution functions of redshift and flux.
        """
        # Normalization
        rho_mat = self.rho_mat
        rho_sum = np.sum(rho_mat)
        rho_norm = rho_mat / rho_sum
        # probability distribution of redshift
        pdf_z = np.sum(rho_norm, axis=0)
        pdf_lumo = np.sum(rho_norm, axis=1)
        # Cumulative function
        cdf_z = np.zeros(pdf_z.shape)
        cdf_lumo = np.zeros(pdf_lumo.shape)
        for i in range(len(pdf_z)):
            cdf_z[i] = np.sum(pdf_z[:i])
        for i in range(len(pdf_lumo)):
            cdf_lumo[i] = np.sum(pdf_lumo[:i])

        return cdf_z, cdf_lumo

    def get_lumo_redshift(self):
        """
        Randomly generate redshif and luminosity at ref frequency using
        the CDF functions.

        Paramaters
        ----------
        df_z, cdf_lumo: np.ndarray
            Cumulative distribution functions of redshift and flux.
        zbin,lumobin: np.ndarray
            Bins of redshif and luminosity.

        Returns
        -------
        z: float
            Redshift.
        lumo: au.W/Hz/sr
            Luminosity.
         """
        # Uniformlly generate random number in interval [0,1]
        rnd_z = np.random.uniform(0, 1)
        rnd_lumo = np.random.uniform(0, 1)
        # Get redshift
        dist_z = np.abs(self.cdf_z - rnd_z)
        idx_z = np.where(dist_z == dist_z.min())
        z = self.zbin[idx_z[0]]
        # Get luminosity
        dist_lumo = np.abs(self.cdf_lumo - rnd_lumo)
        idx_lumo = np.where(dist_lumo == dist_lumo.min())
        lumo = 10 ** self.lumobin[idx_lumo[0]]

        return float(z), float(lumo)

    def gen_single_ps(self):
        """
        Generate single point source, and return its data as a list.
        """
        # Redshift and luminosity
        self.z, self.lumo = self.get_lumo_redshift()
        # angular diameter distance
        self.param = PixelParams(self.z)
        self.dA = self.param.dA
        # W/Hz/Sr to Jy
        dA = self.dA * 3.0856775814671917E+22  # Mpc to meter
        self.lumo = self.lumo / dA**2 / (10.0**-24)  # [Jy]
        # Position
        x = np.random.uniform(0, 1)
        self.lat = (np.arccos(2 * x - 1) / np.pi * 180 - 90)   # [deg]
        self.lon = np.random.uniform(0, np.pi * 2) / np.pi * 180  # [deg]
        # Area
        npix = hp.nside2npix(self.nside)
        self.area = 4 * np.pi / npix  # [sr]

        ps_list = [self.z, self.dA, self.lumo, self.lat, self.lon, self.area]
        return ps_list

    def gen_catalog(self):
        """
        Generate num_ps of point sources and save them into a csv file.
        """
        # Init
        ps_table = np.zeros((self.num_ps, self.nCols))
        for x in range(self.num_ps):
            ps_table[x, :] = self.gen_single_ps()

        # Transform into Dataframe
        self.ps_catalog = pd.DataFrame(ps_table, columns=self.columns,
                                       index=list(range(self.num_ps)))

    def save_as_csv(self):
        """Save the catalog"""
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        pattern = "{prefix}.csv"
        filename = pattern.format(prefix=self.prefix)

        # save to csv
        if self.save:
            file_name = os.path.join(self.output_dir, filename)
            self.ps_catalog.to_csv(file_name)

        return file_name
