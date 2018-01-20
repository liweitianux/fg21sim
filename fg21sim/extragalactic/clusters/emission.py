# Copyright (c) 2017-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Calculate the synchrotron emission for a given relativistic electron
spectrum, e.g., derived for the simulated radio halos.

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

.. [you1998]
   You 1998
   The Radiation Mechanisms in Astrophysics, 2nd Edition, Beijing
   Sec.4.2.3, p.187
"""

import logging
from functools import lru_cache

import numpy as np
import scipy.special
from scipy import integrate, interpolate

from ...share import COSMO
from ...utils.convert import Fnu_to_Tb
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)


logger = logging.getLogger(__name__)


def _interp_sync_kernel(xmin=1e-3, xmax=10.0, xsample=256):
    """
    Sample the synchrotron kernel function at the specified X
    positions and make an interpolation, to optimize the speed
    when invoked to calculate the synchrotron emissivity.

    WARNING
    -------
    Do NOT simply bound the synchrotron kernel within the specified
    [xmin, xmax] range, since it decreases as a power law of index
    1/3 at the left end, and decreases exponentially at the right end.
    Bounding it with interpolation will cause the synchrotron emissivity
    been *overestimated* on the higher frequencies.

    Parameters
    ----------
    xmin, xmax : float, optional
        The lower and upper cuts for the kernel function.
        Default: [1e-3, 10.0]
    xsample : int, optional
        Number of samples within [xmin, xmax] used to do interpolation.

    Returns
    -------
    F_interp : function
        The interpolated kernel function ``F(x)``.
    """
    xx = np.logspace(np.log10(xmin), np.log10(xmax), num=xsample)
    Fxx = [xp * integrate.quad(lambda t: scipy.special.kv(5/3, t),
                               a=xp, b=np.inf)[0]
           for xp in xx]
    F_interp = interpolate.interp1d(xx, Fxx, kind="quadratic",
                                    bounds_error=True, assume_sorted=True)
    return F_interp


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
    # The interpolated synchrotron kernel function ``F(x)`` within
    # the specified range.
    # NOTE: See the *WARNING* above.
    F_xmin = 1e-3
    F_xmax = 10.0
    F_xsample = 256
    F_interp = _interp_sync_kernel(F_xmin, F_xmax, F_xsample)

    def __init__(self, gamma, n_e, B):
        self.gamma = np.asarray(gamma)
        self.n_e = np.asarray(n_e)
        self.B = B  # [uG]

    @property
    @lru_cache()
    def B_gauss(self):
        """
        Magnetic field in unit of [G] (i.e., Gauss)
        """
        return self.B * 1e-6  # [uG] -> [G]

    @property
    @lru_cache()
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
            The angles between the electron velocity and the magnetic field,
            the pitch angle.
            Unit: [rad]

        Returns
        -------
        nu_c : `~numpy.ndarray`
            Critical frequencies
            Unit: [MHz]
        """
        nu_c = 1.5 * gamma**2 * np.sin(theta) * self.frequency_larmor
        return nu_c

    @classmethod
    def F(cls, x):
        """
        Synchrotron kernel function using interpolation to improve speed.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Points where to calculate the kernel function values.
            NOTE: X values will be bounded, e.g., within [1e-5, 20]

        Returns
        -------
        y : `~numpy.ndarray`
            Calculated kernel function values.

        References: Ref.[you1998]
        """
        x = np.array(x, ndmin=1)
        y = np.zeros(x.shape)
        idx = (x >= cls.F_xmin) & (x <= cls.F_xmax)
        y[idx] = cls.F_interp(x[idx])
        # Left end: power law of index 1/3
        idx = (x < cls.F_xmin)
        A = cls.F_interp(cls.F_xmin)
        y[idx] = A * (x[idx] / cls.F_xmin)**(1/3)
        # Right end: exponentially decrease
        idx = (x > cls.F_xmax)
        y[idx] = (0.5*np.pi * x[idx])**0.5 * np.exp(-x[idx])
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

        The pitch angles of electrons w.r.t. the magnetic field are assumed
        to be ``pi/2``, which maybe a good simplification.

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
        nu_c = self.frequency_crit(self.gamma)

        frequencies = np.array(frequencies, ndmin=1)
        syncem = np.zeros(shape=frequencies.shape)
        for i, freq in enumerate(frequencies):
            logger.debug("Calculating emissivity at %.2f [MHz]" % freq)
            kernel = self.F(freq / nu_c)
            # Integrate over energy ``gamma`` in logarithmic grid
            syncem[i] = j_coef * integrate.simps(
                self.n_e*kernel*self.gamma, x=np.log(self.gamma))

        if len(syncem) == 1:
            return syncem[0]
        else:
            return syncem


class HaloEmission:
    """
    Calculate the synchrotron emission of a (giant) radio halo.

    Parameters
    ----------
    gamma : 1D `~numpy.ndarray`
        The Lorentz factors γ of the electron spectrum.
    n_e : 1D `~numpy.ndarray`
        The electron spectrum (w.r.t. Lorentz factors γ).
        Unit: [cm^-3]
    B : float
        The magnetic field strength.
        Unit: [uG]
    radius : float, optional
        The radio halo radius.
        Required to calculate the power.
        Unit: [kpc]
    redshift : float, optional
        The redshift to the radio halo.
        Required to calculate the flux, which also requires ``radius``.
    """
    def __init__(self, gamma, n_e, B, radius=None, redshift=None):
        self.gamma = np.asarray(gamma)
        self.n_e = np.asarray(n_e)
        self.B = B
        self.radius = radius
        self.redshift = redshift

    @property
    def angular_radius(self):
        """
        The angular radius of the radio halo.
        Unit: [arcsec]
        """
        if self.redshift is None:
            raise RuntimeError("parameter 'redshift' is required")
        if self.radius is None:
            raise RuntimeError("parameter 'radius' is required")

        DA = COSMO.DA(self.redshift) * 1e3  # [Mpc] -> [kpc]
        theta = self.radius / DA  # [rad]
        return theta * AUC.rad2arcsec

    @property
    def volume(self):
        """
        The halo volume.
        Unit: [kpc^3]
        """
        if self.radius is None:
            raise RuntimeError("parameter 'radius' is required")

        return (4*np.pi/3) * self.radius**3

    def calc_emissivity(self, frequencies):
        """
        Calculate the synchrotron emissivity for the derived electron
        spectrum.

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the synchrotron emissivity.
            Unit: [MHz]

        Returns
        -------
        emissivity : float, or 1D `~numpy.ndarray`
            The calculated synchrotron emissivity at each specified
            frequency.
            Unit: [erg/s/cm^3/Hz]
        """
        syncem = SynchrotronEmission(gamma=self.gamma, n_e=self.n_e, B=self.B)
        emissivity = syncem.emissivity(frequencies)
        return emissivity

    def calc_power(self, frequencies, emissivity=None):
        """
        Calculate the halo synchrotron power (i.e., power *emitted* per
        unit frequency) by assuming the emissivity is uniform throughout
        the halo volume.

        NOTE
        ----
        The calculated power (a.k.a. spectral luminosity) is in units of
        [W/Hz] which is common in radio astronomy, instead of [erg/s/Hz].
            1 [W] = 1e7 [erg/s]

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the synchrotron power.
            Unit: [MHz]
        emissivity : float, or 1D `~numpy.ndarray`, optional
            The synchrotron emissivity at the input frequencies.
            If not provided, then invoke above ``calc_emissivity()``
            method to calculate them.
            Unit: [erg/s/cm^3/Hz]

        Returns
        -------
        power : float, or 1D `~numpy.ndarray`
            The calculated synchrotron power at each input frequency.
            Unit: [W/Hz]
        """
        frequencies = np.asarray(frequencies)
        if emissivity is None:
            emissivity = self.calc_emissivity(frequencies=frequencies)
        else:
            emissivity = np.asarray(emissivity)
        power = emissivity * (self.volume * AUC.kpc2cm**3)  # [erg/s/Hz]
        power *= 1e-7  # [erg/s/Hz] -> [W/Hz]
        return power

    def calc_flux(self, frequencies):
        """
        Calculate the synchrotron flux density (i.e., power *observed*
        per unit frequency) of the halo, with k-correction considered.

        NOTE
        ----
        The *k-correction* must be applied to the flux density (Sν) or
        specific luminosity (Lν) because the redshifted object is emitting
        flux in a different band than that in which you are observing.
        And the k-correction depends on the spectrum of the object in
        question.  For any other spectrum (i.e., νLν != const.), the flux
        density Sv is related to the specific luminosity Lv by:
            Sν = (1+z) Lν(1+z) / (4π DL^2),
        where
        * Lν(1+z): specific luminosity emitting at frequency ν(1+z),
        * DL: luminosity distance to the object at redshift z.

        Reference: Ref.[hogg1999],Eq.(22)

        Returns
        -------
        flux : float, or 1D `~numpy.ndarray`
            The calculated flux density w.r.t. each input frequency.
            Unit: [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]
        """
        if self.redshift is None:
            raise RuntimeError("parameter 'redshift' is required")

        freqz = np.asarray(frequencies) * (1+self.redshift)
        power = self.calc_power(freqz)  # [W/Hz]
        DL = COSMO.DL(self.redshift) * AUC.Mpc2m  # [m]
        flux = 1e26 * (1+self.redshift) * power / (4*np.pi * DL*DL)  # [Jy]
        return flux

    def calc_brightness_mean(self, frequencies, flux=None, pixelsize=None):
        """
        Calculate the mean surface brightness (power observed per unit
        frequency and per unit solid angle) expressed in *brightness
        temperature* at the specified frequencies.

        NOTE
        ----
        If the solid angle that the object extends is smaller than the
        specified pixel area, then is is assumed to have size of 1 pixel.

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the mean brightness temperature
            Unit: [MHz]
        flux : float, or 1D `~numpy.ndarray`, optional
            The flux density w.r.t. each input frequency.
            Unit: [Jy]
        pixelsize : float, optional
            The pixel size of the output simulated sky image.
            If not provided, then invoke above ``calc_flux()`` method to
            calculate them.
            Unit: [arcsec]

        Returns
        -------
        Tb : float, or 1D `~numpy.ndarray`
            The mean brightness temperature at each frequency.
            Unit: [K] <-> [Jy/pixel]
        """
        frequencies = np.asarray(frequencies)
        if flux is None:
            flux = self.calc_flux(frequencies=frequencies)  # [Jy]
        else:
            flux = np.asarray(flux)
        omega = np.pi * self.angular_radius**2  # [arcsec^2]
        if pixelsize and (omega < pixelsize**2):
            omega = pixelsize ** 2  # [arcsec^2]
            logger.warning("Halo size < 1 pixel; force to be 1 pixel!")

        Tb = Fnu_to_Tb(flux, omega, frequencies)  # [K]
        return Tb
