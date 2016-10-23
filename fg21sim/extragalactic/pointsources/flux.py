# Copyright (c) 2016 Zhixian MA <zxma_sjtu@qq.com>
# MIT license

"""
A class to calculate the brightness temperature of point sources is
defined, based on the works of Wang et al. and Willman et al.

[1] Wang J, Xu H, Gu J, et al. How to Identify and Separate Bright Galaxy
    Clusters from the Low-frequency Radio Sky?[J]. Astrophysical Journal, 2010,
    723(1):620-633.

[2]  Wilman R J, Miller L, Jarvis M J, et al.
    A semi-empirical simulation of the extragalactic radio continuum sky for
    next generation radio telescopes[J]. Monthly Notices of the Royal
    Astronomical Society, 2008, 388(3):1335–1348.

"""
import numpy as np

# Defination of class
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
        |  RQ AGN	|  3   |
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

    def __init__(self, freq=150, ClassType=1):
        # Frequency
        self.freq = freq
        # ClassType = ClassType
        self.ClassType = ClassType

    def gen_spec(self):
        # generate the spectrum
        # Use IF-THEN to replace SWITCH-CASE
        # reference flux at 151MHz, see Willman et al's work
        self.I_151 = 10**(np.random.uniform(-4, -3))
        # Clac flux
        if self.ClassType == 1:
            spec = (self.freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 2:
            spec = (self.freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 3:
            spec = (self.freq / 151e6)**(-0.7) * self.I_151
        elif self.ClassType == 4:
            spec_lobe = (self.freq / 151e6)**-0.75 * self.I_151
            a0 = (np.log10(self.I_151) - 0.7 * np.log10(151e6) +
                0.29 * np.log10(151e6) * np.log10(151e6))
            lgs = (a0 + 0.7 * np.log10(self.freq) - 0.29 *
                np.log10(self.freq) * np.log10(self.freq))
            spec_core = 10**lgs
            spec = np.array([spec_core, spec_lobe])
        elif self.ClassType == 5:
            spec_lobe = (self.freq / 151e6)**-0.75 * self.I_151
            spec_hotspot = (self.freq / 151e6)**-0.75 * self.I_151
            a0 = (np.log10(self.I_151) - 0.7 * np.log10(151e6) +
                0.29 * np.log10(151e6) * np.log10(151e6))
            lgs = (a0 + 0.7 * np.log10(self.freq) - 0.29 *
                np.log10(self.freq) * np.log10(self.freq))
            spec_core = 10**lgs
            spec = np.array([spec_core, spec_lobe, spec_hotspot])

        return spec

    # calc_Tb
    def calc_Tb(self, area):
        # light speed
        c = 2.99792458e8
        # ?
        kb = 1.38e-23
        # flux in Jy
        flux_in_Jy = self.gen_spec()
        Omegab = area  # [sr]

        Sb = flux_in_Jy * 1e-26 / Omegab
        flux_pixel = Sb / 2 / self.freq / self.freq * c * c / kb

        return flux_pixel


