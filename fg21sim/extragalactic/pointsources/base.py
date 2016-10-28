# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license

"""
Simulation the radio emissions from various types of point sources (PS)

Currently supported PS types:

* Star-forming (SF) galaxies
* Star-bursting (SB) galaxies
* Radio-quiet AGNs
* Radio-loud AGNs: Fanaroff-Riley type I (FRI)
* Radio-loud AGNs: Fanaroff-Riley type II (FRII)

References
----------
.. [Wilman2008]
   Wilman et al.,
   "A semi-empirical simulation of the extragalactic radio continuum
   sky for next generation radio telescopes",
   2008, MNRAS, 388, 1335-1348,
   http://adsabs.harvard.edu/abs/2008MNRAS.388.1335W

.. [Jelic2008]
   Jelic et al.,
   "Foreground simulations for the LOFAR-Epoch of Reionization
   Experiment",
   2008, MNRAS, 389, 1319-1335,
   http://adsabs.harvard.edu/abs/2008MNRAS.389.1319W
"""


import os
import logging

import numpy as np
import astropy.units as au
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import healpy as hp

from ...utils.random import spherical_uniform


logger = logging.getLogger(__name__)


class BasePointSource:
    """
    Base class for point sources simulation

    FIXME: rewrite this doc string!

    Attributes
    ----------
    z: float
        Redshift, z ~ U(0,20)
    dA: au.Mpc;
        Angular diameter distance, which is calculated according to the
        cosmology constants.
    lumo: au.Jy;
        Luminosity at the reference frequency
    lat: au.deg;
        The latitude in the spherical coordinate system
    lon: au.deg;
        The longitude in the spherical coordinate system
    area: au.sr;
        Angular size/area of the point sources
    """
    # Identifier of the PS component
    comp_id = "extragalactic/pointsources"
    # ID of this PS type
    ps_type = None
    # Name of this PS type
    name = None

    def __init__(self, configs):
        # configures
        self.configs = configs
        # FIXME: get rid of this `columns`
        # Basic PS catalog columns
        self.columns = ['z', 'dA (Mpc)', 'luminosity (Jy)', 'Lat (deg)',
                        'Lon (deg)', 'Area (sr)']
        self._set_configs()

    def _set_configs(self):
        """Load the configs and set the corresponding class attributes."""
        # Configurations shared between all supported PS types
        comp = self.comp_id
        self.resolution = self.configs.getn(comp+"/resolution") * au.arcmin
        self.catalog_pattern = self.configs.getn(comp+"/catalog_pattern")
        self.save = self.configs.getn(comp+"/save")
        self.output_dir = self.configs.get_path(comp+"/output_dir")
        # Specific configurations for this PS type
        pstype = "/".join([self.comp_id, self.ps_type])
        self.number = self.configs.getn(pstype+"/number")
        self.prefix2 = self.configs.getn(pstype+"/prefix2")
        self.save2 = self.configs.getn(pstype+"/save2")
        #
        self.use_float = self.configs.getn("output/use_float")
        self.clobber = self.configs.getn("output/clobber")
        self.nside = self.configs.getn("common/nside")
        self.npix = hp.nside2npix(self.nside)
        # Cosmology model
        self.H0 = self.configs.getn("cosmology/H0")
        self.OmegaM0 = self.configs.getn("cosmology/OmegaM0")
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.OmegaM0)
        #
        logger.info("Loaded and set up configurations")

    def calc_number_density(self):
        raise NotImplementedError("Sub-class must implement this method!")

    # FIXME: directly sample from joint distribution of z and flux/luminosity
    def calc_cdf(self):
        """
        Calculate cumulative distribution functions for simulating of
        samples with corresponding reshift and luminosity.

        Parameters
        ----------
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
        # FIXME: use `cumsum()`
        for i in range(len(pdf_z)):
            cdf_z[i] = np.sum(pdf_z[:i])
        for i in range(len(pdf_lumo)):
            cdf_lumo[i] = np.sum(pdf_lumo[:i])

        return cdf_z, cdf_lumo

    # FIXME: get rid of self.* !
    def get_lumo_redshift(self):
        """
        Randomly generate redshif and luminosity at ref frequency using
        the CDF functions.

        Parameters
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

    # FIXME: Get rid of this function!
    def gen_single_ps(self):
        """Generate the information of one single point source"""
        # Redshift and luminosity
        z, lumo = self.get_lumo_redshift()
        # angular diameter distance
        DA = self.cosmo.angular_diameter_distance(z).to(au.Mpc).value
        # [ W/Hz/sr ] => [ Jy ]
        DA_m = DA * 3.0856775814671917E+22  # Mpc to meter
        lumo = lumo / DA_m**2 / 1e-24  # [Jy]
        # Position
        theta, phi = spherical_uniform()
        glon = np.rad2deg(phi)
        glat = 90.0 - np.rad2deg(theta)
        # Area
        area = 4 * np.pi / self.npix  # [sr]
        ps_data = [z, DA_m, lumo, glat, glon, area]
        return ps_data

    # FIXME: The catalog should be simulated on the column based, not on
    #        single PS based, which slows the speed!
    def gen_catalog(self):
        """
        Generate num_ps of point sources and save them into a csv file.
        """
        shape = (self.number, len(self.columns))
        catalog = np.zeros(shape)
        for i in range(shape[0]):
            catalog[i, :] = self.gen_single_ps()
        self.catalog = pd.DataFrame(catalog, columns=self.columns)
        logger.info("Done simulate the catalog")

    def save_catalog(self):
        """Save the catalog"""
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            logger.info("Created output directory: %s" % self.output_dir)
        # Append the prefix for the specific PS type
        if self.prefix2 is not None:
            prefix = "_".join([self.prefix, self.prefix2])
        filepath = self._make_filepath(pattern=self.catalog_pattern,
                                       prefix=prefix)
        # Save catalog data
        if os.path.exists(filepath):
            if self.clobber:
                logger.warning("Remove existing catalog file: %s" % filepath)
                os.remove(filepath)
            else:
                raise OSError("Output file already exists: %s" % filepath)
        self.catalog.to_csv(filepath, header=True, index=False)
        logger.info("Save clusters catalog in use to: {0}".format(filepath))

    def output(self):
        raise NotImplementedError("TODO")

    def _make_filepath(self, pattern=None, **kwargs):
        """Make the path of output file according to the filename pattern
        and output directory loaded from configurations.

        Parameters
        ----------
        pattern : str, optional
            Specify the filename (without the extension) pattern in string
            format template syntax.  If not specified, then use
            ``self.filename_pattern``.
        **kwargs : optional
            Other named parameters used to format the filename pattern.

        Returns
        -------
        filepath : str
            The full path to the output file (with directory and extension).
        """
        data = {
            "prefix": self.prefix,
        }
        data.update(kwargs)
        if pattern is None:
            pattern = self.filename_pattern
        filename = pattern.format(**data)
        filetype = self.configs.getn("output/filetype")
        if filetype == "fits":
            filename += ".fits"
        else:
            raise NotImplementedError("unsupported filetype: %s" % filetype)
        filepath = os.path.join(self.output_dir, filename)
        return filepath
