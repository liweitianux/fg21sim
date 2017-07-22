# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate (giant) radio halo originating from the last/ most recent
cluster-cluster major merger event, following the "statistical
magneto-turbulent model" proposed by [cassano2005]_, but with many
modifications and simplifications.

References
----------
.. [brunetti2011]
   Brunetti & Lazarian 2011, MNRAS, 410, 127
   http://adsabs.harvard.edu/abs/2011MNRAS.410..127B

.. [cassano2005]
   Cassano & Brunetti 2005, MNRAS, 357, 1313
   http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C

.. [cassano2006]
   Cassano, Brunetti & Setti, 2006, MNRAS, 369, 1577
   http://adsabs.harvard.edu/abs/2006MNRAS.369.1577C

.. [cassano2012]
   Cassano et al. 2012, A&A, 548, A100
   http://adsabs.harvard.edu/abs/2012A%26A...548A.100C

.. [donnert2013]
   Donnert 2013, AN, 334, 615
   http://adsabs.harvard.edu/abs/2013AN....334..515D

.. [donnert2014]
   Donnert & Brunetti 2014, MNRAS, 443, 3564
   http://adsabs.harvard.edu/abs/2014MNRAS.443.3564D

.. [sarazin1999]
   Sarazin 1999, ApJ, 520, 529
   http://adsabs.harvard.edu/abs/1999ApJ...520..529S
"""

import logging

import numpy as np

from . import helper
from .solver import FokkerPlanckSolver
from ...utils import cosmo
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)


logger = logging.getLogger(__name__)


class RadioHalo:
    """
    Simulate the extended radio halo emission from galaxy cluster
    experiencing on-going/recent merger.

    Description
    -----------
    1. Calculate the merger crossing time (t_cross; ~1 Gyr);
    2. Calculate the diffusion coefficient (Dpp) from the systematic
       acceleration timescale (tau_acc; ~0.1 Gyr).  The acceleration
       diffusion is assumed to have an action time ~ t_cross (i.e.,
       only during merger crossing), and then been disabled (i.e.,
       only radiation and ionization losses later);
    3. Assume the electrons are constantly injected and has a power-law
       energy spectrum;
    4. Determine the initial electron density

    after that, calculate the electron acceleration and time evolution
    by solving the Fokker-Planck equation; and finally derive the radio
    emission from the electron spectra.

    Parameters
    ----------
    M0 : float
        Cluster virial mass at redshift z0
        Unit: [Msun]
    z0 : float
        Redshift from where to simulate former merging history.
    """
    def __init__(self, M0, z0):
        self.M0 = M0
        self.z0 = z0

    def calc_electron_spectrum(self, zbegin=None, zend=None, n0_e=None):
        """
        Calculate the relativistic electron spectrum by solving the
        Fokker-Planck equation.

        Parameters
        ----------
        zbegin : float, optional
            The redshift from where to solve the Fokker-Planck equation.
            Default: ``self.zmax``.
        zend : float, optional
            The redshift where to stop solving the Fokker-Planck equation.
            Default: ``self.z0``.
        n0_e : 1D `~numpy.ndarray`, optional
            The initial electron number distribution.
            Unit: [cm^-3].
            Default: accumulated constant-injected electrons until zbegin.

        Returns
        -------
        gamma : `~numpy.ndarray`
            The Lorentz factor grid adopted for solving the equation.
        n_e : `~numpy.ndarray`
            The solved electron spectrum at ``zend``.
            Unit: [cm^-3]
        """
        if zbegin is None:
            tstart = cosmo.age(self.zmax)
        else:
            tstart = cosmo.age(zbegin)
        if zend is None:
            tstop = cosmo.age(self.z0)
        else:
            tstop = cosmo.age(zend)

        fpsolver = FokkerPlanckSolver(
            xmin=self.gamma_min, xmax=self.gamma_max,
            x_np=self.gamma_np,
            tstep=self.time_step,
            f_advection=self.fp_advection,
            f_diffusion=self.fp_diffusion,
            f_injection=self.fp_injection,
            buffer_np=self.buffer_np,
        )
        gamma = fpsolver.x
        if n0_e is None:
            # Accumulated constant-injected electrons until ``tstart``.
            n_inj = np.array([self.fp_injection(gm) for gm in gamma])
            n0_e = n_inj * tstart
        n_e = fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
        return (gamma, n_e)

    def _z_end(self, z_begin, time):
        """
        Calculate the ending redshift from ``z_begin`` after elapsing
        ``time``.

        Parameters
        ----------
        z_begin : float
            Beginning redshift
        time : float
            Elapsing time (unit: Gyr)
        """
        t_begin = cosmo.age(z_begin)  # [Gyr]
        t_end = t_begin + time
        if t_end >= cosmo.age(0):
            z_end = 0.0
        else:
            z_end = cosmo.redshift(t_end)
        return z_end

    def fp_injection(self, p, t=None):
        """
        Electron injection term for the Fokker-Planck equation.

        The injected electrons are assumed to have a power-law spectrum
        and a constant injection rate.

        Qe(p) = Ke * (p/pmin)**(-s)
        Ke = ((s-2)*eta_e) * (e_th/(pmin*c)) / (t0*pmin)

        Parameters
        ----------
        p : float
            Electron momentum (unit: mec), i.e., Lorentz factor
        t : None
            Currently a constant injection rate is assumed, therefore
            this parameter is not used.  Keep it for the consistency
            with other functions.

        Returns
        -------
        Qe : float
            Current electron injection rate at specified energy (p).
            Unit: [cm^-3 Gyr^-1 mec^-1]

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eqs.(31-33)
        """
        if not hasattr(self, "_electron_injection_rate"):
            e_th = self.e_thermal  # [erg/cm^3]
            age = cosmo.age(self.z0)
            term1 = (self.injection_index-2) * self.eta_e
            term2 = e_th / (self.pmin * self.mec * AC.c)  # [cm^-3]
            term3 = 1.0 / (age * self.pmin)  # [Gyr^-1 mec^-1]
            Ke = term1 * term2 * term3
            self._electron_injection_rate = Ke
        else:
            Ke = self._electron_injection_rate
        Qe = Ke * (p/self.pmin) ** (-self.injection_index)
        return Qe

    def fp_diffusion(self, p, t):
        """
        Diffusion term/coefficient for the Fokker-Planck equation.

        Parameters
        ----------
        p : float
            Electron momentum (unit: mec), i.e., Lorentz factor
        t : float
            Current time when solving the equation (unit: Gyr)

        Returns
        -------
        Dpp : float
            Diffusion coefficient
            Unit: [mec^2/Gyr]

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(36)
        [2] Donnert 2013, AN, 334, 615
            http://adsabs.harvard.edu/abs/2013AN....334..515D
            Eq.(15)
        """
        z = cosmo.redshift(t)
        chi = self._coef_acceleration(z)  # [Gyr^-1]
        # NOTE: Cassano & Brunetti's formula misses a factor of 2.
        Dpp = chi * p**2 / 4  # [mec^2/Gyr]
        return Dpp

    def fp_advection(self, gamma, t):
        """
        Advection term/coefficient for the Fokker-Planck equation,
        which describes a systematic tendency for upward or downard
        drift of particles.

        This term is also called the "generalized cooling function"
        by [donnert2014], which includes all relevant energy loss
        functions and the energy gain function due to turbulence.

        Returns
        -------
        advection : float
            Advection coefficient, describing the energy loss/gain rate.
            Unit: [Gyr^-1]
        """
        advection = (abs(self._loss_ion(gamma, t)) +
                     abs(self._loss_rad(gamma, t)) -
                     (self.fp_diffusion(gamma, t) * 2 / gamma))
        return advection

    def _loss_ion(self, gamma, t):
        """
        Energy loss through ionization and Coulomb collisions.

        Parameters
        ----------
        gamma : float
            The Lorentz factor of electrons
        t : float
            The cosmic time/age
            Unit: [Gyr]

        Returns
        -------
        loss : float
            The energy loss rate
            Unit: [Gyr^-1]

        References
        ----------
        Ref.[sarazin1999],Eq.(9)
        """
        z = cosmo.redshift(t)
        mass = self._mass(t)
        n_th = helper.density_number_thermal(mass, z)
        coef = -1.20e-12 * AUC.Gyr2s  # [Gyr^-1]
        loss = coef * n_th * (1 + np.log(gamma/n_th) / 75)
        return loss

    def _loss_rad(self, gamma, t):
        """
        Energy loss via synchrotron emission and inverse Compton
        scattering off the CMB photons.

        References
        ----------
        Ref.[sarazin1999],Eq.(6,7)
        """
        z = cosmo.redshift(t)
        coef = -1.37e-20 * AUC.Gyr2s  # [Gyr^-1]
        loss = (coef * gamma**2 *
                ((self.magnetic_field/3.25)**2 + (1+z)**4))
        return loss
