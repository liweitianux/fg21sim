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
import scipy.special
from scipy import integrate
from scipy import interpolate

from ...utils import COSMO
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)
from ...utils.convert import Fnu_to_Tb_fast


logger = logging.getLogger(__name__)


class SynchrotronEmission:
    """
    Calculate the synchrotron emission from a given population
    of electrons.

    Parameters
    ----------
    gamma : `~numpy.ndarray`
        The Lorentz factors of electrons.
    n_e : `~numpy.ndarray`
        Electron number density spectrum.
        Unit: [cm^-3]
    z : float
        Redshift of the cluster/halo been observed/simulated.
    B : float
        The assumed uniform magnetic field within the cluster ICM.
        Unit: [uG]
    radius : float
        The radius of the halo, within which the uniform magnetic field
        and electron distribution are assumed.
        Unit: [kpc]
    """
    def __init__(self, gamma, n_e, z, B, radius):
        self.gamma = np.asarray(gamma)
        self.n_e = np.asarray(n_e)
        self.z = z
        self.B = B  # [uG]
        self.radius = radius  # [kpc]

    @property
    def frequency_larmor(self):
        """
        Electron Larmor frequency:
            ν_L = e * B / (2*π * m0 * c) = e * B / (2*π * mec)

        Unit: [MHz]
        """
        coef = AC.e / (2*np.pi * AU.mec)  # [Hz/G]
        coef *= 1e-12  # [Hz/G] -> [MHz/uG]
        nu = coef * self.B  # [MHz]
        return nu

    def frequency_crit(self, gamma, theta=np.pi/2):
        """
        Synchrotron critical frequency.

        Critical frequency:
            ν_c = (3/2) * γ^2 * sin(θ) * ν_L

        Parameters
        ----------
        gamma : float, or `~numpy.ndarray`
            Electron Lorentz factors γ
        theta : float, or `~numpy.ndarray`, optional
            The angles between the electron velocity and the magnetic field.
            Unit: [rad]

        Returns
        -------
        nu : float, or `~numpy.ndarray`
            Critical frequencies
            Unit: [MHz]
        """
        nu_L = self.frequency_larmor
        nu = (3/2) * gamma**2 * np.sin(theta) * nu_L
        return nu

    @staticmethod
    def F(x):
        """
        Synchrotron kernel function.

        NOTE
        ----
        Use interpolation to optimize the speed, also avoid instabilities
        near the lower end (e.g., x < 1e-5).
        Interpolation also helps vectorize this function for easier calling.

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
        # The lower and upper cuts
        xmin = 1e-5
        xmax = 20.0
        # Number of samples within [xmin, xmax]
        # NOTE: this kernel function is quiet smooth and slow-varying.
        nsamples = 128
        # Make an interpolation
        x_interp = np.logspace(np.log10(xmin), np.log10(xmax),
                               num=nsamples)
        F_interp = [
            xp * integrate.quad(lambda t: scipy.special.kv(5/3, t),
                                a=xp, b=np.inf)[0]
            for xp in x_interp
        ]
        func_interp = interpolate.interp1d(x_interp, F_interp,
                                           kind="quadratic")

        x = np.array(x)  # Make a copy!
        x[x < xmin] = xmin
        x[x > xmax] = xmax
        y = func_interp(x)
        return y

    def emissivity(self, nu):
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
        nu : float
            Frequency where to calculate the emissivity.
            Unit: [MHz]

        Returns
        -------
        j_nu : float
            Synchrotron emissivity at frequency ``nu``.
            Unit: [erg/s/cm^3/Hz]
        """
        # Ignore the zero angle
        theta = np.linspace(0, np.pi/2, num=len(self.gamma))[1:]
        theta_grid, gamma_grid = np.meshgrid(theta, self.gamma)
        nu_c = self.frequency_crit(gamma_grid, theta_grid)
        Fv = np.vectorize(self.F, otypes=[np.float])
        x = nu / nu_c
        kernel = Fv(x)

        # 2D samples over width to do the integration
        s2d = kernel * np.outer(self.n_e, np.sin(theta)**2)
        # Integrate over ``theta`` (the last axis)
        s1d = integrate.simps(s2d, x=theta)
        # Integrate over energy ``gamma`` in logarithmic grid
        j_nu = integrate.simps(s1d*self.gamma, np.log(self.gamma))

        coef = np.sqrt(3) * AC.e**3 * self.B / AC.c
        j_nu *= coef
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
