#!/usr/bin/env python3
#
# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
Basic parameters to be used in point sources simulation.
References
------------
[1] Angular diameter distance,
     https://en.wikipedia.org/wiki/Angular_diameter_distance
[2] FlatLambdaCDM,
     {http://docs.astropy.org/en/stable/api/astropy.cosmology.
     FlatLambdaCDM.html#astropy.cosmology.FlatLambdaCDM}
"""

# astropy related modules
import astropy.units as au    # units for astronomy physics
from astropy.cosmology import FlatLambdaCDM


class PixelParams():
    """
    A class to transform cosmology distance to angles or pixels.
    Parameters
    ------------
    H0: float
        The hubble constant at z = 0, whose unit is km/s/Mpc
    Om0: float
        The total matter density.
    ang_res: float
        Angular resolution, i.e. degree per pixel. (May be useless)
    ang_total: list
        Total angles of the simulated sky region,whose unit is degree (\deg)
    z : float
        Redshift
    scale: float
        The real object scale.
    Example
    ------------
    >>> PixelParams = PixelParams(img_size=(1024,1024),ang_total=(5,5))
    """
    # Hubble constant at z = 0
    H0 = 71.0
    # Omega0, total matter density
    Om0 = 0.27
    # Redshift
    z = 0
    # Cosmology calculator
    cosmo = 0
    # angular diameter distance, [Mpc]
    dA = 0 * au.Mpc
    # angular resolution
    ang_res = au.rad * 0

    def __init__(self, z=0):
        self.z = z
        self.cosmo = FlatLambdaCDM(H0=self.H0, Om0=self.Om0)

        # Angle per pixel
        # self.ang_res = ang_res.to(au.rad)  # [rac]

        # angular diameter distance, [Mpc]
        self.dA = self.cosmo.angular_diameter_distance(self.z)

    def get_angle(self, scale=1*au.Mpc):
        """
        Input real object scale, and output the respect observed
        angle, and pixels.
        """
        ang = scale / self.dA # [rac]
        # ang_pix = ang /self.ang_res # [pix]

        return ang

    def get_scale(self, ang=au.rad * 1):
        """
        Input real observed scale, and output the respect
        real object scale, and pixels.
        """
        scale = ang * self.dA #[Mpc]
        #scale_pix = ang / self.ang_res

        return scale
