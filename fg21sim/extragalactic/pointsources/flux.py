# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
A class to calculate the brightness temperature of point sources is
defined, based on the works of Wang et al. and Willman et al.

[1] Wang J et al.,
    "How to Identify and Separate Bright Galaxy Clusters from the
    Low-frequency Radio Sky?",
    2010,ApJ,723,620-633.,
    http://adsabs.harvard.edu/abs/2010ApJ...723..620W.
[2] Wilman et al.,
    "A semi-empirical simulation of the extragalactic radio continuum
    sky for next generation radio telescopes",
    2008,MNRAS,388,1335-1348.,
    http://adsabs.harvard.edu/abs/2008MNRAS.389.1335W.
"""

import numpy as np
import astropy.constants as ac
import astropy.units as au

class Flux:
    """
    To calculate the flux and surface brightness of the point sources
    accordingly to its type and frequency

    Parameters
    ----------
    freq: float
        The frequency
    ClassType: int
        The type of point source, which is default as 1.
        | ClassType | Code |
        |:---------:|:----:|
        |    SF     |  1   |
        |    SB     |  2   |
        |  RQ AGN   |  3   |
        |   FRI     |  4   |
        |   FRII    |  5   |


    Functions
    ---------
    genspec:
        Generate the spectrum of the source at frequency freq.
    calc_Tb:
        Calculate the average surface brightness, the area of the source
        should be inputed.
    """

    def __init__(self, freq=150, class_type=1):
        self.freq = freq
        self.class_type = class_type

    def gen_spec(self):
        """ Generate the spectrum """
        # Init
        freq_ref = 151e6
        # Todo
        self.I_151 = 10**(np.random.uniform(-4, -3))
        # Clac flux
        if self.class_type == 1:
            spec = (self.freq / freq_ref)**(-0.7) * self.I_151
        elif self.class_type == 2:
            spec = (self.freq / freq_ref)**(-0.7) * self.I_151
        elif self.class_type == 3:
            spec = (self.freq / freq_ref)**(-0.7) * self.I_151
        elif self.class_type == 4:
            spec_lobe = (self.freq / freq_ref)**-0.75 * self.I_151
            a0 = (np.log10(self.I_151) - 0.7 * np.log10(freq_ref) +
                0.29 * np.log10(freq_ref) * np.log10(freq_ref))
            lgs = (a0 + 0.7 * np.log10(self.freq) - 0.29 *
                np.log10(self.freq) * np.log10(self.freq))
            spec_core = 10**lgs
            spec = np.array([spec_core, spec_lobe])
        elif self.class_type == 5:
            spec_lobe = (self.freq / freq_ref)**-0.75 * self.I_151
            spec_hotspot = (self.freq / freq_ref)**-0.75 * self.I_151
            a0 = (np.log10(self.I_151) - 0.7 * np.log10(freq_ref) +
                0.29 * np.log10(freq_ref) * np.log10(freq_ref))
            lgs = (a0 + 0.7 * np.log10(self.freq) - 0.29 *
                np.log10(self.freq) * np.log10(self.freq))
            spec_core = 10**lgs
            spec = np.array([spec_core, spec_lobe, spec_hotspot])

        return spec

    # calc_Tb
    def calc_Tb(self, area):
        """ Calculate average surface brightness of the point source"""
        # light speed
        c = ac.c.cgs.value
        # ?
        kb = ac.k_B.cgs.value
        # flux in Jy
        flux = self.gen_spec()
        Omegab = area  # [sr]

        Sb = (flux * au.Jy).to(au.Unit("J/m2")) / Omegab
        flux_pixel = (0.5*Sb) /(self.freq **2) * (c**2) / kb

        return flux_pixel.value
