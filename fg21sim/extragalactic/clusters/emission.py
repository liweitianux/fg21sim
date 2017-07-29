# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Calculate the synchrotron emission and inverse Compton emission
for simulated radio halos.

References
----------
.. [cassano2005]
   Cassano & Brunetti 2005, MNRAS, 357, 1313
   http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
   Appendix.C

.. [era2016]
   Condon & Ransom 2016
   Essential Radio Astronomy
   https://science.nrao.edu/opportunities/courses/era/
   Chapter.5
"""

import logging

import numpy as np
import scipy.special
from scipy import integrate
from scipy import interpolate

from ...utils.units import (Units as AU, Constants as AC)


logger = logging.getLogger(__name__)


class SynchrotronEmission:
    """
    Calculate the synchrotron emissivity from a given population
    of electrons.

    Parameters
    ----------
    gamma : `~numpy.ndarray`
        The Lorentz factors of electrons.
    n_e : `~numpy.ndarray`
        Electron number density spectrum.
        Unit: [cm^-3]
    B : float
        The assumed uniform magnetic field within the cluster ICM.
        Unit: [uG]
    """
    def __init__(self, gamma, n_e, B):
        self.gamma = np.asarray(gamma)
        self.n_e = np.asarray(n_e)
        self.B = B  # [uG]

    @property
    def B_gauss(self):
        """
        Magnetic field in unit of [G] (i.e., gauss)
        """
        return self.B * 1e-6  # [uG] -> [G]

    @property
    def frequency_larmor(self):
        """
        Electron Larmor frequency (a.k.a. gyro frequency):
            ν_L = e * B / (2*π * m0 * c) = e * B / (2*π * mec)
        =>  ν_L [MHz] = 2.8 * B [G]

        Unit: [MHz]
        """
        nu_larmor = AC.e * self.B_gauss / (2*np.pi * AU.mec)  # [Hz]
        return nu_larmor * 1e-6  # [Hz] -> [MHz]

    def frequency_crit(self, gamma, theta=np.pi/2):
        """
        Synchrotron critical frequency.

        Critical frequency:
            ν_c = (3/2) * γ^2 * sin(θ) * ν_L

        Parameters
        ----------
        gamma : `~numpy.ndarray`
            Electron Lorentz factors γ
        theta : `~numpy.ndarray`, optional
            The angles between the electron velocity and the magnetic field.
            Unit: [rad]

        Returns
        -------
        nu_c : `~numpy.ndarray`
            Critical frequencies
            Unit: [MHz]
        """
        nu_c = 1.5 * gamma**2 * np.sin(theta) * self.frequency_larmor
        return nu_c

    def F(self, x):
        """
        Synchrotron kernel function.

        NOTE
        ----
        * Use interpolation to optimize the speed, also avoid instabilities
          near the lower end (e.g., x < 1e-5).
        * Interpolation also helps vectorize this function for easier calling.
        * Cache the interpolation results, since this function will be called
          multiple times for each frequency.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Points where to calculate the kernel function values.
            NOTE: X values will be bounded, e.g., within [1e-5, 20]

        Returns
        -------
        y : `~numpy.ndarray`
            Calculated kernel function values.
        """
        if not hasattr(self, "_F_interp"):
            # Interpolate the kernel function and cache the results
            #
            # The lower and upper cuts
            xmin = 1e-5
            xmax = 20.0
            # Number of samples within [xmin, xmax]
            # NOTE: this kernel function is quiet smooth and slow-varying.
            nsamples = 128
            # Make an interpolation
            xx = np.logspace(np.log10(xmin), np.log10(xmax), num=nsamples)
            Fxx = [xp * integrate.quad(lambda t: scipy.special.kv(5/3, t),
                                       a=xp, b=np.inf)[0]
                   for xp in xx]
            self._F_interp = interpolate.interp1d(xx, Fxx, kind="quadratic")

        x = np.array(x)  # Make a copy!
        x[x < xmin] = xmin
        x[x > xmax] = xmax
        y = self._F_interp(x)
        return y

    def emissivity(self, frequencies):
        """
        Calculate the synchrotron emissivity (power emitted per volume
        and per frequency) at the requested frequency.

        NOTE
        ----
        Since ``self.gamma`` and ``self.n_e`` are sampled on a logarithmic
        grid, we integrate over ``ln(gamma)`` instead of ``gamma`` directly:
            I = int_gmin^gmax f(g) d(g)
              = int_ln(gmin)^ln(gmax) f(g) g d(ln(g))

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the synchrotron emissivity.
            Unit: [MHz]

        Returns
        -------
        syncem : float, or 1D `~numpy.ndarray`
            The calculated synchrotron emissivity at each specified
            frequency.
            Unit: [erg/s/cm^3/Hz]
        """
        j_coef = np.sqrt(3) * AC.e**3 * self.B_gauss / AU.mec2
        # Ignore the zero angle
        theta = np.linspace(0, np.pi/2, num=len(self.gamma))[1:]
        theta_grid, gamma_grid = np.meshgrid(theta, self.gamma)
        nu_c = self.frequency_crit(gamma_grid, theta_grid)
        # 2D grid of ``n_e(gamma) * sin^2(theta)``
        nsin2 = np.outer(self.n_e, np.sin(theta)**2)

        frequencies = np.array(frequencies, ndmin=1)
        syncem = np.zeros(shape=frequencies.shape)
        for i, freq in zip(range(len(frequencies)), frequencies):
            logger.debug("Calc synchrotron emissivity at %.2f [MHz]" % freq)
            kernel = self.F(freq / nu_c)
            # 2D samples over width to do the integration
            s2d = kernel * nsin2
            # Integrate over ``theta`` (the last axis)
            s1d = integrate.simps(s2d, x=theta)
            # Integrate over energy ``gamma`` in logarithmic grid
            syncem[i] = j_coef * integrate.simps(s1d*self.gamma,
                                                 np.log(self.gamma))

        if len(syncem) == 1:
            return syncem[0]
        else:
            return syncem
