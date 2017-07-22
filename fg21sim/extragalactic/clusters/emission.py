# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Calculate the synchrotron emission and inverse Compton emission
for simulated radio halos.

References
----------
[1] Cassano & Brunetti 2005, MNRAS, 357, 1313
    http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
    Appendix.C
"""

import logging

import numpy as np
import scipy.integrate
import scipy.special

from ...utils import COSMO
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)
from ...utils.convert import Fnu_to_Tb_fast


logger = logging.getLogger(__name__)


class SynchrotronEmission:
    """
    Calculate the synchrotron emission spectrum from a given population
    of electrons.

    Parameters
    ----------
    B : float
        The assumed uniform magnetic field of the galaxy cluster.
        Unit: [uG]
    p : `~numpy.ndarray`
        The momentum grid adopted when solving the Fokker-Planck equation.
        Unit: [mec]
    n_e : `~numpy.ndarray`
        Electron spectrum by solving the Fokker-Planck equation.
        Unit: [cm^-3 mec^-1]
    radius, float
        The radius of the galaxy cluster/halo, within which the uniform
        magnetic field and electron distribution are assumed.
        Unit: [kpc]
    z : float
        Redshift of the galaxy cluster/halo
    """
    def __init__(self, B, p, n_e, radius, z):
        self.B = B  # [uG]
        self.p = p
        self.n_e = n_e
        self.z = z
        self.radius = radius  # [kpc]

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

    @staticmethod
    def F(x):
        """
        Synchrotron kernel function.
        """
        val = x * scipy.integrate.quad(lambda t: scipy.special.kv(5/3, t),
                                       a=x, b=np.inf)[0]
        return val

    def emissivity(self, nu):
        """
        Calculate the synchrotron emissivity (power emitted per volume
        and per frequency) at the requested frequency.

        Parameters
        ----------
        nu : float
            Frequency where to calculate the emissivity.
            Unit: [MHz]

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
        func_p = np.zeros(self.p.shape)
        for i in range(len(self.p)):
            # Integrate over ``theta``
            func_p[i] = scipy.integrate.quad(
                lambda t: func(t, self.p[i], self.n_e[i]),
                a=0, b=np.pi/2)[0]
        # Integrate over ``p``
        j_nu = coef * scipy.integrate.trapz(func_p, self.p)
        return j_nu

    def power(self, nu):
        """
        Calculate the synchrotron power (power emitted per frequency)
        at the requested frequency.

        Returns
        -------
        P_nu : float
            Synchrotron power at frequency ``nu``.
            Unit: [erg/s/Hz]
        """
        r_cm = self.radius * AUC.kpc2cm
        volume = (4.0/3.0) * np.pi * r_cm**3
        P_nu = self.emissivity(nu) * volume
        return P_nu

    def flux(self, nu):
        """
        Calculate the synchrotron flux (power observed per frequency)
        at the requested frequency.

        Returns
        -------
        F_nu : float
            Synchrotron flux at frequency ``nu``.
            Unit: [Jy] = 1e-23 [erg/s/cm^2/Hz]
        """
        DL = COSMO.DL(self.z) * AUC.Mpc2cm  # [cm]
        P_nu = self.power(nu)
        F_nu = 1e23 * P_nu / (4*np.pi * DL*DL)  # [Jy]
        return F_nu

    def brightness(self, nu, pixelsize):
        """
        Calculate the synchrotron surface brightness (power observed
        per frequency and per solid angle) at the specified frequency.

        NOTE
        ----
        If the radio halo has solid angle less than the pixel area, then
        it is assumed to have solid angle of 1 pixel.

        Parameters
        ----------
        pixelsize : float
            The pixel size of the output simulated sky image
            Unit: [arcsec]

        Returns
        -------
        Tb : float
            Synchrotron surface brightness at frequency ``nu``.
            Unit: [K] <-> [Jy/pixel]
        """
        DA = COSMO.DL(self.z) * AUC.Mpc2cm  # [cm]
        radius = self.radius * AUC.kpc2cm  # [cm]
        omega = (np.pi * radius**2 / DA**2) * AUC.rad2deg**2  # [deg^2]
        pixelarea = (pixelsize * AUC.arcsec2deg) ** 2  # [deg^2]
        if omega < pixelarea:
            omega = pixelarea
        F_nu = self.flux(nu)
        Tb = Fnu_to_Tb_fast(F_nu, omega, nu)
        return Tb
