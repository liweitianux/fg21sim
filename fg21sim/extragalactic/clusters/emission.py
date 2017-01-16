# Copyright (c) 2017 Weitian LI <liweitianux@live.com>
# MIT license

"""
Calculate the synchrotron emission for simulated radio halos.
"""

import logging

import numpy as np
import scipy.integrate
import scipy.special

from .units import (Units as AU, Constants as AC)


logger = logging.getLogger(__name__)


class SynchrotronEmission:
    """
    Calculate the synchrotron emission from a given population
    of electrons.

    References
    ----------
    [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
        http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
        Appendix.C
    """
    def __init__(self, B):
        # Uniform magnetic field strength
        self.B = B  # [uG]

    @property
    def frequency_larmor(self):
        """
        Electron Larmor frequency:
            ν_L = e * B / (2*π * m0 * c) = e * B / (2*π * mec)

        Unit: MHz
        """
        coef = AC.e / (2*np.pi * AU.mec)  # [Hz/G]
        coef *= 1e-12  # [MHz/uG]
        nu = coef * self.B  # [MHz]
        return nu

    def frequency_crit(self, p, theta=np.pi/2):
        """
        Synchrotron critical frequency.

        Critical frequency:
            ν_c = (3/2) * γ^2 * sin(θ) * ν_L

        Parameters
        ----------
        p : float
            Electron momentum (unit: mec), i.e., Lorentz factor γ
        theta : float, optional
            The angle between the electron velocity and the magnetic field.
            (unit: radian)

        Returns
        -------
        nu : float
            Critical frequency, unit: MHz
        """
        nu_L = self.frequency_larmor
        nu = (3/2) * p**2 * np.sin(theta) * nu_L
        return nu

    def F(self, x):
        """
        Synchrotron kernel function.
        """
        val = x * scipy.integrate.quad(lambda t: scipy.special.kv(5/3, t),
                                       a=x, b=np.inf)[0]
        return val

    def emissivity(self, nu, p, n_e):
        """
        Calculate the synchrotron emissivity (power emitted per volume
        and per frequency) at the specified frequency from the given
        electron number & energy distribution

        Parameters
        ----------
        nu : float
            Frequency (unit: MHz) where to calculate the emissivity.
        p : `~numpy.ndarray`
            The momentum grid adopted when solving the Fokker-Planck equation.
            Unit: [mec]
        n_e : `~numpy.ndarray`
            Electron spectrum by solving the Fokker-Planck equation.
            Unit: [cm^-3 mec^-1]

        Returns
        -------
        j_nu : float
            Synchrotron emissivity at frequency ``nu``.
            Unit: [erg/s/cm^3/Hz]
        """
        def func(theta, _p, _n_e):
            nu_c = self.frequency_crit(_p, theta)
            x = nu / nu_c
            return (np.sin(theta)**2 * _n_e * self.F(x))

        coef = np.sqrt(3) * AC.e**3 * self.B / AC.c  # multiplied a [mec]
        func_p = np.zeros(p.shape)
        for i in range(len(p)):
            # Integrate over ``theta``
            func_p[i] = scipy.integrate.quad(
                lambda t: func(t, p[i], n_e[i]),
                a=0, b=np.pi/2)[0]
        # Integrate over ``p``
        j_nu = coef * scipy.integrate.trapz(func_p, p)
        return j_nu
