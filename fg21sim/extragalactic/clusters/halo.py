# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate (giant) radio halo originating from the last/recent
cluster-cluster major merger event, following the "statistical
magneto-turbulent model" proposed by [cassano2005]_, but with many
modifications and simplifications.

References
----------
.. [brunetti2011]
   Brunetti & Lazarian 2011, MNRAS, 410, 127
   http://adsabs.harvard.edu/abs/2011MNRAS.410..127B

.. [brunetti2016]
   Brunetti 2016, PPCF, 58, 014011
   http://adsabs.harvard.edu/abs/2016PPCF...58a4011B

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

.. [hogg1999]
   Hogg 1999, arXiv:astro-ph/9905116
   http://adsabs.harvard.edu/abs/1999astro.ph..5116H

.. [miniati2015]
   Miniati & Beresnyak 2015, Nature, 523, 59
   http://adsabs.harvard.edu/abs/2015Natur.523...59M

.. [sarazin1999]
   Sarazin 1999, ApJ, 520, 529
   http://adsabs.harvard.edu/abs/1999ApJ...520..529S
"""

import logging
from functools import lru_cache

import numpy as np

from . import helper
from .solver import FokkerPlanckSolver
from .emission import SynchrotronEmission
from ...share import CONFIGS, COSMO
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)
from ...utils.convert import Fnu_to_Tb


logger = logging.getLogger(__name__)


class RadioHalo:
    """
    Simulate the extended radio halo emission from the galaxy cluster
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
       energy spectrum, determine the injection rate by further assuming
       that the total injected electrons has energy of a fraction (eta_e)
       of the ICM total thermal energy;
    4. Set the initial electron density/spectrum be the total injected
       electrons during t_merger time;
    5. Calculate the magnetic field from the cluster total mass (which
       is assumed to be growth linearly from M_main+M_sub to M_obs);
    6. Calculate the energy losses for the coefficients of Fokker-Planck
       equation;
    7. Solve the Fokker-Planck equation to derive the relativistic
       electron spectrum at t_obs (i.e., z_obs);
    8. Calculate the synchrotron emissivity from the derived electron
       spectrum.

    Parameters
    ----------
    M_obs : float
        Cluster virial mass at the current observation (simulation end) time.
        Unit: [Msun]
    z_obs : float
        Redshift of the current observation (simulation end) time.
    M_main, M_sub : float
        The main and sub cluster masses before the (major) merger.
        Unit: [Msun]
    z_merger : float
        The redshift when the (major) merger begins.

    Attributes
    ----------
    fpsolver : `~FokkerPlanckSolver`
        The solver instance to calculate the electron spectrum evolution.
    radius : float
        The halo radius
        Unit: [kpc]
    gamma : 1D float `~numpy.ndarray`
        The Lorentz factors of the adopted logarithmic grid to solve the
        equation.
    electron_spec : 1D float `~numpy.ndarray`
        The derived electron (number density) distribution/spectrum at
        the final time (``zend``), which is set by the methods
        ``self.calc_electron_spectrum()`` or ``self.set_electron_spectrum()``.
        Unit: [cm^-3]
    """
    def __init__(self, M_obs, z_obs, M_main, M_sub, z_merger,
                 configs=CONFIGS):
        self.M_obs = M_obs
        self.z_obs = z_obs
        self.M_main = M_main
        self.M_sub = M_sub
        self.z_merger = z_merger

        self.configs = configs
        self._set_configs()
        self._set_solver()

    def _set_configs(self):
        comp = "extragalactic/halos"
        self.f_lturb = self.configs.getn(comp+"/f_lturb")
        self.f_acc = self.configs.getn(comp+"/f_acc")
        self.eta_turb = self.configs.getn(comp+"/eta_turb")
        self.eta_e = self.configs.getn(comp+"/eta_e")
        self.gamma_min = self.configs.getn(comp+"/gamma_min")
        self.gamma_max = self.configs.getn(comp+"/gamma_max")
        self.gamma_np = self.configs.getn(comp+"/gamma_np")
        self.buffer_np = self.configs.getn(comp+"/buffer_np")
        self.time_step = self.configs.getn(comp+"/time_step")
        self.injection_index = self.configs.getn(comp+"/injection_index")

    def _set_solver(self):
        self.fpsolver = FokkerPlanckSolver(
            xmin=self.gamma_min, xmax=self.gamma_max,
            x_np=self.gamma_np,
            tstep=self.time_step,
            f_advection=self.fp_advection,
            f_diffusion=self.fp_diffusion,
            f_injection=self.fp_injection,
            buffer_np=self.buffer_np,
        )

    @property
    @lru_cache()
    def gamma(self):
        """
        The logarithmic grid adopted for solving the equation.
        """
        return self.fpsolver.x

    @property
    def age_obs(self):
        return COSMO.age(self.z_obs)

    @property
    def age_merger(self):
        return COSMO.age(self.z_merger)

    @property
    def tback_merger(self):
        """
        The time from the observation (``z_obs``) back to the merger
        (``z_merger``).
        """
        return (self.age_obs - self.age_merger)  # [Gyr]

    @property
    @lru_cache()
    def time_crossing(self):
        """
        The time duration of the sub-cluster crossing the main cluster,
        which is also used to approximate the merging time, during which
        the turbulence acceleration is regarded as effective.

        Unit: [Gyr]
        """
        return helper.time_crossing(self.M_main, self.M_sub,
                                    z=self.z_merger)

    @property
    def radius_virial_obs(self):
        """
        The virial radius of the "current" cluster (``M_obs``) at
        ``z_obs``.

        Unit: [kpc]
        """
        return helper.radius_virial(mass=self.M_obs, z=self.z_obs)

    @property
    def radius_virial_main(self):
        """
        The virial radius of the main cluster at ``z_merger``.
        """
        return helper.radius_virial(mass=self.M_main, z=self.z_merger)

    @property
    def radius_virial_sub(self):
        return helper.radius_virial(mass=self.M_sub, z=self.z_merger)

    @property
    @lru_cache()
    def radius(self):
        """
        The estimated radius for the simulated radio halo.

        NOTE
        ----
        The halo radius is assumed to be the virial radius of the falling
        sub-cluster.  See ``helper.radius_halo()`` for more details.

        Unit: [kpc]
        """
        r_halo = helper.radius_halo(self.M_main, self.M_sub,
                                    self.z_merger)
        return r_halo

    @property
    def angular_radius(self):
        """
        The angular radius of the radio halo.

        Unit: [arcsec]
        """
        DA = COSMO.DA(self.z_obs) * 1e3  # [Mpc] -> [kpc]
        theta = self.radius / DA  # [rad]
        return theta * AUC.rad2arcsec

    @property
    def volume(self):
        """
        The halo volume, calculated from the above radius.

        Unit: [kpc^3]
        """
        return (4*np.pi/3) * self.radius**3

    @property
    @lru_cache()
    def magnetic_field(self):
        """
        The magnetic field strength at the simulated observation
        time (i.e., cluster mass of ``self.M_obs``), will be used
        to calculate the synchrotron emissions.

        Unit: [uG]
        """
        return helper.magnetic_field(mass=self.M_obs, z=self.z_obs)

    @property
    @lru_cache()
    def kT_main(self):
        """
        The mean temperature of the main cluster ICM at ``z_merger``
        when the merger begins.

        Unit: [keV]
        """
        return helper.kT_cluster(mass=self.M_main, z=self.z_merger)

    @property
    @lru_cache()
    def kT_sub(self):
        return helper.kT_cluster(mass=self.M_sub, z=self.z_merger)

    @property
    @lru_cache()
    def kT_obs(self):
        """
        The "current" cluster ICM mean temperature at ``z_obs``.
        """
        return helper.kT_cluster(self.M_obs, z=self.z_obs)  # [keV]

    @property
    @lru_cache()
    def Mach_turbulence(self):
        """
        The Mach number of the merger-induced turbulence.

        The turbulence  Mach number:
            Mach_turb = sqrt(<δv>^2) / c_s
                      ≅ sqrt(sqrt(3)/α) * sqrt(η_turb/0.37)
        where:
        c_s is the sound speed,
        α is a parameter ranges about 1.5-3, and we take it as:
            α = 3^(3/2) / 2 ≅ 2.6
        η_turb describes the fraction of thermal energy originating from
        turbulent dissipation, ~0.3.

        Reference: Ref.[miniati2015],Eq.(1)
        """
        alpha = 3**1.5 / 2
        mach = np.sqrt(3**0.5 * self.eta_turb / alpha / 0.37)
        return mach

    @property
    @lru_cache()
    def tau_acceleration(self):
        """
        Calculate the electron acceleration timescale due to turbulent
        waves at the given (cosmic) time, which describes the turbulent
        acceleration efficiency.

        Unit: [Gyr]

        NOTE
        ----
        Generally, the turbulent acceleration timescale is about 0.1 Gyr.
        It is shown that this acceleration timescale depends weakly on
        cluster mass and redshift, therefore, its value is derived at the
        beginning of the merger and assumed to be constant throughout the
        merging period.

        Reference: Ref.[brunetti2016],Eq.(8,9)
        """
        Mach = self.Mach_turbulence
        Rvir = helper.radius_virial(mass=self.M_main, z=self.z_merger)
        cs = helper.speed_sound(self.kT_main)  # [km/s]
        # Turbulence injection scale
        L0 = self.f_lturb * Rvir  # [kpc]
        x = cs*AUC.km2cm / AC.c
        fx = x * (x**4/4 + x*x - (1+2*x*x) * np.log(x) - 5/4)
        term1 = self.f_acc * 2.5 / fx / (Mach/0.5)**4
        term2 = (L0/300) / (cs/1500)
        tau = term1 * term2 / 1000  # [Gyr]
        return tau

    @property
    @lru_cache()
    def injection_rate(self):
        """
        The constant electron injection rate assumed.
        Unit: [cm^-3 Gyr^-1]

        The injection rate is parametrized by assuming that the total
        energy injected in the relativistic electrons during the cluster
        life (e.g., ``age_obs`` here) is a fraction (``self.eta_e``)
        of the total thermal energy of the cluster.

        The electrons are assumed to be injected throughout the cluster
        ICM/volume, i.e., do not restricted inside the halo volume.

        Qe(γ) = Ke * γ^(-s),
        int[ Qe(γ) γ me c^2 ]dγ * t_cluster = eta_e * e_th
        =>
        Ke = [(s-2) * eta_e * e_th * γ_min^(s-2) / (me * c^2 * t_cluster)]

        References
        ----------
        Ref.[cassano2005],Eqs.(31,32,33)
        """
        s = self.injection_index
        e_thermal = helper.density_energy_thermal(self.M_obs, self.z_obs)
        term1 = (s-2) * self.eta_e * e_thermal  # [erg cm^-3]
        term2 = self.gamma_min**(s-2)
        term3 = AU.mec2 * self.age_obs  # [erg Gyr]
        Ke = term1 * term2 / term3  # [cm^-3 Gyr^-1]
        return Ke

    def calc_electron_spectrum(self, zbegin=None, zend=None, n0_e=None):
        """
        Calculate the relativistic electron spectrum by solving the
        Fokker-Planck equation.

        Parameters
        ----------
        zbegin : float, optional
            The redshift from where to solve the Fokker-Planck equation.
            Default: ``self.z_merger``.
        zend : float, optional
            The redshift where to stop solving the Fokker-Planck equation.
            Default: ``self.z_obs``.
        n0_e : 1D `~numpy.ndarray`, optional
            The initial electron number distribution.
            Unit: [cm^-3].
            Default: accumulated constantly injected electrons until zbegin.

        Returns
        -------
        electron_spec : float 1D `~numpy.ndarray`
            The solved electron spectrum at ``zend``.
            Unit: [cm^-3]
        """
        if zbegin is None:
            tstart = COSMO.age(self.z_merger)
        else:
            tstart = COSMO.age(zbegin)
        if zend is None:
            tstop = COSMO.age(self.z_obs)
        else:
            tstop = COSMO.age(zend)
        if n0_e is None:
            # Accumulated constantly injected electrons until ``tstart``.
            n_inj = self.fp_injection(self.gamma)
            n0_e = n_inj * tstart

        # When the evolution time is too short, decrease the time step
        # to improve the results.
        # XXX: is this necessary???
        nstep_min = 20
        if (tstop - tstart) / self.time_step < nstep_min:
            tstep = (tstop - tstart) / nstep_min
            logger.debug("Decreased time step: %g -> %g [Gyr]" %
                         (self.time_step, self.fpsolver.tstep))
            self.fpsolver.tstep = tstep

        self.electron_spec = self.fpsolver.solve(u0=n0_e, tstart=tstart,
                                                 tstop=tstop)
        return self.electron_spec

    def set_electron_spectrum(self, n_e):
        """
        Check the given electron spectrum and set it to the
        ``self.electron_spec``.

        Parameters
        ----------
        n_e : float 1D `~numpy.ndarray`
            The solved electron spectrum at ``zend``.
            Unit: [cm^-3]
        """
        n_e = np.array(n_e)  # make a copy
        if n_e.shape == self.gamma.shape:
            self.electron_spec = n_e
        else:
            raise ValueError("given electron spectrum has wrong shape!")

    def calc_emissivity(self, frequencies, n_e=None, gamma=None):
        """
        Calculate the synchrotron emissivity for the derived electron
        spectrum.

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the synchrotron emissivity.
            Unit: [MHz]
        n_e : 1D `~numpy.ndarray`, optional
            The electron spectrum (w.r.t. Lorentz factors γ).
            If not provided, then used the cached ``self.electron_spec``
            solved above.
            Unit: [cm^-3]
        gamma : 1D `~numpy.ndarray`, optional
            The Lorentz factors γ of the electron spectrum.
            If not provided, then used ``self.gamma``.

        Returns
        -------
        emissivity : float, or 1D `~numpy.ndarray`
            The calculated synchrotron emissivity at each specified
            frequency.
            Unit: [erg/s/cm^3/Hz]
        """
        if n_e is None:
            n_e = self.electron_spec
        if gamma is None:
            gamma = self.gamma
        syncem = SynchrotronEmission(gamma=gamma, n_e=n_e,
                                     B=self.magnetic_field)
        emissivity = syncem.emissivity(frequencies)
        return emissivity

    def calc_power(self, frequencies, emissivity=None, **kwargs):
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
        **kwargs : optional arguments, i.e., ``n_e`` and ``gamma``

        Returns
        -------
        power : float, or 1D `~numpy.ndarray`
            The calculated synchrotron power at each input frequency.
            Unit: [W/Hz]
        """
        frequencies = np.asarray(frequencies)
        if emissivity is None:
            emissivity = self.calc_emissivity(frequencies=frequencies,
                                              **kwargs)
        else:
            emissivity = np.asarray(emissivity)
            if emissivity.shape != frequencies.shape:
                raise ValueError("input 'frequencies' and 'emissivity' "
                                 "do not match")
        power = emissivity * (self.volume * AUC.kpc2cm**3)  # [erg/s/Hz]
        power *= 1e-7  # [erg/s/Hz] -> [W/Hz]
        return power

    def calc_flux(self, frequencies, **kwargs):
        """
        Calculate the synchrotron flux density (i.e., power *observed*
        per unit frequency) of the halo, with k-correction considered.

        NOTE
        ----
        The *k-correction* must be applied to the flux density (Sν) or
        specific luminosity (Lν) because the redshifted object is emitting
        flux in a different band than that in which you are observing.
        And the k-correction depends on the spectrum of the object in
        question.  For any other spectrum (i.e., vLv != const.), the flux
        density Sv is related to the specific luminosity Lv by:
            Sv = (1+z) L_v(1+z) / (4π DL^2),
        where
        * L_v(1+z) is the specific luminosity emitting at frequency v(1+z),
        * DL is the luminosity distance to the object at redshift z.

        Reference: Ref.[hogg1999],Eq.(22)

        Parameters
        ----------
        frequencies : float, or 1D `~numpy.ndarray`
            The frequencies where to calculate the flux density.
            Unit: [MHz]
        **kwargs : optional arguments, i.e., ``n_e`` and ``gamma``

        Returns
        -------
        flux : float, or 1D `~numpy.ndarray`
            The calculated flux density w.r.t. each input frequency.
            Unit: [Jy] = 1e-23 [erg/s/cm^2/Hz] = 1e-26 [W/m^2/Hz]
        """
        z = self.z_obs
        freqz = np.asarray(frequencies) * (1+z)
        power = self.calc_power(freqz, **kwargs)  # [W/Hz]
        DL = COSMO.DL(self.z_obs) * AUC.Mpc2m  # [m]
        flux = 1e26 * (1+z) * power / (4*np.pi * DL*DL)  # [Jy]
        return flux

    def calc_brightness_mean(self, frequencies, flux=None, pixelsize=None,
                             **kwargs):
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
        **kwargs : optional arguments, i.e., ``n_e`` and ``gamma``

        Returns
        -------
        Tb : float, or 1D `~numpy.ndarray`
            The mean brightness temperature at each frequency.
            Unit: [K] <-> [Jy/pixel]
        """
        frequencies = np.asarray(frequencies)
        if flux is None:
            flux = self.calc_flux(frequencies=frequencies, **kwargs)  # [Jy]
        else:
            flux = np.asarray(flux)
            if flux.shape != frequencies.shape:
                raise ValueError("input 'frequencies' and 'flux' do not match")

        omega = np.pi * self.angular_radius**2  # [arcsec^2]
        if pixelsize and (omega < pixelsize**2):
            omega = pixelsize ** 2  # [arcsec^2]
            logger.warning("Object size < 1 pixel; force to be 1 pixel!")

        Tb = Fnu_to_Tb(flux, omega, frequencies)  # [K]
        return Tb

    def fp_injection(self, gamma, t=None):
        """
        Electron injection (rate) term for the Fokker-Planck equation.

        NOTE
        ----
        The injected electrons are assumed to have a power-law spectrum
        and a constant injection rate.

        Qe(γ) = Ke * γ^(-s),
        Ke: constant injection rate

        Parameters
        ----------
        gamma : float, or float 1D `~numpy.ndarray`
            Lorentz factors of electrons
        t : None
            Currently a constant injection rate is assumed, therefore
            this parameter is not used.  Keep it for the consistency
            with other functions.

        Returns
        -------
        Qe : float, or float 1D `~numpy.ndarray`
            Current electron injection rate at specified energies (gamma).
            Unit: [cm^-3 Gyr^-1]

        References
        ----------
        Ref.[cassano2005],Eqs.(31,32,33)
        """
        Ke = self.injection_rate  # [cm^-3 Gyr^-1]
        Qe = Ke * gamma**(-self.injection_index)
        return Qe

    def fp_diffusion(self, gamma, t):
        """
        Diffusion term/coefficient for the Fokker-Planck equation.

        The diffusion is directly related to the electron acceleration
        which is described by the ``tau_acc`` acceleration timescale
        parameter.

        NOTE
        ----
        Considering that the turbulence acceleration is a 2nd-order Fermi
        process, it has only an effective acceleration time of several 1e8
        years.  Therefore, the turbulence is assumed to only accelerate
        the electrons during the merging period, i.e., the acceleration
        timescale is set to be infinite after "t_merger + time_cross".

        However, a zero diffusion coefficient may lead to unstable/wrong
        results, so constrain the acceleration timescale to be a large
        enough but finite number (e.g., 10 Gyr).

        Parameters
        ----------
        gamma : float, or float 1D `~numpy.ndarray`
            The Lorentz factors of electrons
        t : float
            Current (cosmic) time when solving the equation
            Unit: [Gyr]

        Returns
        -------
        diffusion : float, or float 1D `~numpy.ndarray`
            Diffusion coefficients
            Unit: [Gyr^-1]

        References
        ----------
        Ref.[donnert2013],Eq.(15)
        """
        if t < (self.age_merger + self.time_crossing):
            tau_acc = self.tau_acceleration  # [Gyr]
        else:
            # The large enough timescale to avoid unstable results
            tau_acc = 10.0  # [Gyr]
        diffusion = gamma**2 / 4 / tau_acc
        return diffusion

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
        advection : float, or float 1D `~numpy.ndarray`
            Advection coefficients, describing the energy loss/gain rates.
            Unit: [Gyr^-1]
        """
        advection = (abs(self._loss_ion(gamma, t)) +
                     abs(self._loss_rad(gamma, t)) -
                     (self.fp_diffusion(gamma, t) * 2 / gamma))
        return advection

    def _mass(self, t):
        """
        Calculate the main cluster mass at the given (cosmic) time.

        NOTE
        ----
        We assume that the main cluster grows (i.e., gains mass) linearly
        in time from (M_main, z_merge) to (M_obs, z_obs).

        Parameters
        ----------
        t : float
            The (cosmic) time/age.
            Unit: [Gyr]

        Returns
        -------
        mass : float
            The mass of the main cluster.
            Unit: [Msun]
        """
        t_merger = self.age_merger
        rate = (self.M_obs - self.M_main) / (self.age_obs - t_merger)
        mass = rate * (t - t_merger) + self.M_main
        return mass

    def _loss_ion(self, gamma, t):
        """
        Energy loss through ionization and Coulomb collisions.

        Parameters
        ----------
        gamma : float, or float 1D `~numpy.ndarray`
            The Lorentz factors of electrons
        t : float
            The cosmic time/age
            Unit: [Gyr]

        Returns
        -------
        loss : float, or float 1D `~numpy.ndarray`
            The energy loss rates
            Unit: [Gyr^-1]

        References
        ----------
        Ref.[sarazin1999],Eq.(9)
        """
        z = COSMO.redshift(t)
        mass = self._mass(t)
        n_th = helper.density_number_thermal(mass, z)  # [cm^-3]
        loss = -3.79e4 * n_th * (1 + np.log(gamma/n_th) / 75)
        return loss

    def _loss_rad(self, gamma, t):
        """
        Energy loss via synchrotron emission and inverse Compton
        scattering off the CMB photons.

        References
        ----------
        Ref.[sarazin1999],Eq.(6,7)
        """
        z = COSMO.redshift(t)
        mass = self._mass(t)  # [Msun]
        B = helper.magnetic_field(mass=mass, z=z)  # [uG]
        loss = -4.32e-4 * gamma**2 * ((B/3.25)**2 + (1+z)**4)
        return loss
