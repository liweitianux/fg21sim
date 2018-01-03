# Copyright (c) 2017-2018 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate (giant) radio halos originating from the recent merger
events, which generate cluster-wide turbulence and accelerate the
primary (i.e., fossil) relativistic electrons to high energies to
be synchrotron-bright.  This *turbulence re-acceleration* model
is currently most widely accepted to explain the (giant) radio halos.

The simulation method is somewhat based on the statistical (Monte
Carlo) method proposed by [cassano2005]_, but with extensive
modifications and improvements.

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

.. [hogg1999]
   Hogg 1999, arXiv:astro-ph/9905116
   http://adsabs.harvard.edu/abs/1999astro.ph..5116H

.. [miniati2015]
   Miniati 2015, ApJ, 800, 60
   http://adsabs.harvard.edu/abs/2015ApJ...800...60M

.. [pinzke2017]
   Pinzke, Oh & Pfrommer 2017, MNRAS, 465, 4800
   http://adsabs.harvard.edu/abs/2017MNRAS.465.4800P

.. [sarazin1999]
   Sarazin 1999, ApJ, 520, 529
   http://adsabs.harvard.edu/abs/1999ApJ...520..529S
"""

import logging
from functools import lru_cache

import numpy as np

from . import helper
from .solver import FokkerPlanckSolver
from ...share import CONFIGS, COSMO
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)


logger = logging.getLogger(__name__)


class RadioHalo:
    """
    Simulate the diffuse (giant) radio halo emission for a galaxy
    cluster experiencing on-going/recent merger.

    Description
    -----------
    1. Calculate the turbulence persistence time (tau_turb; ~<1 Gyr);
    2. Calculate the diffusion coefficient (D_pp) from the systematic
       acceleration timescale (tau_acc; ~0.1 Gyr).  The acceleration
       diffusion is assumed to have an action time ~ tau_turb (i.e.,
       only during turbulence persistence), and then is disabled (i.e.,
       only radiation and ionization losses later);
    3. Assume the electrons are constantly injected and has a power-law
       energy spectrum, determine the injection rate by further assuming
       that the total injected electrons has energy of a fraction (eta_e)
       of the ICM total thermal energy;
    4. Set the electron density/spectrum be the accumulated electrons
       injected during t_merger time, then evolve it for time_init period
       considering only losses and constant injection, in order to derive
       an approximately steady electron spectrum for following use;
    5. Calculate the magnetic field from the cluster total mass (which
       is assumed to be growth linearly from M_main to M_obs);
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
    # Component name
    compID = "extragalactic/halos"
    name = "giant radio halos"

    def __init__(self, M_obs, z_obs, M_main, M_sub, z_merger,
                 configs=CONFIGS):
        self.M_obs = M_obs
        self.z_obs = z_obs
        self.M_main = M_main
        self.M_sub = M_sub
        self.z_merger = z_merger

        self._set_configs(configs)
        self._set_solver()

    def _set_configs(self, configs):
        comp = self.compID
        self.configs = configs
        self.f_acc = configs.getn(comp+"/f_acc")
        self.f_lturb = configs.getn(comp+"/f_lturb")
        self.zeta_ins = configs.getn(comp+"/zeta_ins")
        self.eta_turb = configs.getn(comp+"/eta_turb")
        self.eta_e = configs.getn(comp+"/eta_e")
        self.x_cr = configs.getn(comp+"/x_cr")
        self.gamma_min = configs.getn(comp+"/gamma_min")
        self.gamma_max = configs.getn(comp+"/gamma_max")
        self.gamma_np = configs.getn(comp+"/gamma_np")
        self.buffer_np = configs.getn(comp+"/buffer_np")
        if self.buffer_np == 0:
            self.buffer_np = None
        self.time_step = configs.getn(comp+"/time_step")
        self.time_init = configs.getn(comp+"/time_init")
        self.injection_index = configs.getn(comp+"/injection_index")

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
    def age_begin(self):
        """
        The cosmic time when the merger begins.
        Unit: [Gyr]
        """
        return COSMO.age(self.z_merger)

    @property
    def tback_merger(self):
        """
        The time from the observation (``z_obs``) back to the beginning
        of the merger (``z_merger``).
        """
        return (self.age_obs - self.age_begin)  # [Gyr]

    @property
    @lru_cache()
    def time_turbulence(self):
        """
        The time duration the merger-induced turbulence persists, which
        is used to approximate the effective turbulence acceleration
        timescale.

        Unit: [Gyr]
        """
        return helper.time_turbulence(self.M_main, self.M_sub,
                                      z=self.z_merger, configs=self.configs)

    @property
    def mach_turbulence(self):
        """
        The turbulence Mach number determined from its velocity dispersion.
        """
        cs = helper.speed_sound(self.kT_main)  # [km/s]
        v_turb = self._velocity_turb()  # [km/s]
        return v_turb / cs

    @property
    def radius_virial_obs(self):
        """
        The virial radius of the "current" cluster (``M_obs``) at
        ``z_obs``.

        Unit: [kpc]
        """
        return helper.radius_virial(mass=self.M_obs, z=self.z_obs)

    @property
    @lru_cache()
    def radius(self):
        """
        The estimated radius for the simulated radio halo.
        Unit: [kpc]
        """
        return helper.radius_halo(self.M_obs, self.z_obs, configs=self.configs)

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
    def B_obs(self):
        """
        The magnetic field strength at the simulated observation
        time (i.e., cluster mass of ``self.M_obs``), will be used
        to calculate the synchrotron emissions.

        Unit: [uG]
        """
        return helper.magnetic_field(mass=self.M_obs, z=self.z_obs,
                                     configs=self.configs)

    @property
    @lru_cache()
    def kT_obs(self):
        """
        The ICM mean temperature of the cluster at ``z_obs``.
        Unit: [keV]
        """
        return helper.kT_cluster(self.M_obs, z=self.z_obs,
                                 configs=self.configs)

    @property
    @lru_cache()
    def kT_main(self):
        """
        The mean temperature of the main cluster ICM at ``z_merger``
        when the merger begins.

        Unit: [keV]
        """
        return helper.kT_cluster(mass=self.M_main, z=self.z_merger,
                                 configs=self.configs)

    @property
    @lru_cache()
    def tau_acceleration(self):
        """
        Calculate the electron acceleration timescale due to turbulent
        waves, which describes the turbulent acceleration efficiency.
        The turbulent acceleration timescale has order of ~0.1 Gyr.

        Here we consider the turbulence cascade mode through scattering
        in the high-β ICM mediated by plasma instabilities (firehose,
        mirror) rather than Coulomb scattering.  Therefore, the fast modes
        damp by TTD (transit time damping) on relativistic rather than
        thermal particles, and the diffusion coefficient is given by:
            D_pp = (2*p^2 * ζ / η_e) * k_L * <v_turb^2>^2 / c_s^3
        where:
            ζ: efficiency factor for the effectiveness of plasma instabilities
            η_e: relative energy density of cosmic rays (injected relativistic
                 electrons??)
            k_L = 2π/L: turbulence injection scale
            v_turb: turbulence velocity dispersion
            c_s: sound speed
        Thus the acceleration timescale is:
            τ_acc = p^2 / (4*D_pp)
                  = (η_e * c_s^3 * L) / (16π * ζ * <v_turb^2>^2)

        Unit: [Gyr]

        Reference
        ---------
        * Ref.[pinzke2017],Eq.(37)
        * Ref.[miniati2015],Eq.(29)
        """
        R_vir = helper.radius_virial(mass=self.M_main, z=self.z_merger)
        L = self.f_lturb * R_vir  # [kpc]
        cs = helper.speed_sound(self.kT_main)  # [km/s]
        v_turb = self._velocity_turb()  # [km/s]
        tau = (self.x_cr * cs**3 * L /
               (16*np.pi * self.zeta_ins * v_turb**4))  # [s kpc/km]
        tau *= AUC.s2Gyr * AUC.kpc2km  # [Gyr]
        tau *= self.f_acc  # custom tune parameter
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
        int[ Qe(γ) γ me c^2 ]dγ * t_cluster = η_e * e_th
        =>
        Ke = [(s-2) * η_e * e_th * γ_min^(s-2) / (me * c^2 * t_cluster)]

        References
        ----------
        Ref.[cassano2005],Eqs.(31,32,33)
        """
        s = self.injection_index
        e_th = helper.density_energy_thermal(self.M_obs, self.z_obs,
                                             configs=self.configs)
        term1 = (s-2) * self.eta_e * e_th  # [erg cm^-3]
        term2 = self.gamma_min**(s-2)
        term3 = AU.mec2 * self.age_obs  # [erg Gyr]
        Ke = term1 * term2 / term3  # [cm^-3 Gyr^-1]
        return Ke

    @property
    def electron_spec_init(self):
        """
        The electron spectrum at ``age_begin`` to be used as the initial
        condition for the Fokker-Planck equation.

        This initial electron spectrum is derived from the accumulated
        electron spectrum injected throughout the ``age_begin`` period,
        by solving the same Fokker-Planck equation, but only considering
        energy losses and constant injection, evolving for a period of
        ``time_init`` in order to obtain an approximately steady electron
        spectrum.

        Units: [cm^-3]
        """
        # Accumulated electrons constantly injected until ``age_begin``
        n_inj = self.fp_injection(self.gamma)
        n0_e = n_inj * (self.age_begin - self.time_init)

        logger.debug("Derive the initial electron spectrum ...")
        # NOTE: subtract ``time_step`` to avoid the acceleration at the
        #       last step at ``age_begin``.
        tstart = self.age_begin - self.time_init - self.time_step
        tstop = self.age_begin - self.time_step
        # Use a bigger time step to save time
        self.fpsolver.tstep = 3 * self.time_step
        n_e = self.fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
        # Restore the original time step
        self.fpsolver.tstep = self.time_step
        return n_e

    def calc_electron_spectrum(self, tstart=None, tstop=None, n0_e=None):
        """
        Calculate the relativistic electron spectrum by solving the
        Fokker-Planck equation.

        Parameters
        ----------
        tstart : float, optional
            The (cosmic) time from when to solve the Fokker-Planck equation
            for relativistic electrons evolution.
            Default: ``self.age_begin``.
            Unit: [Gyr]
        tstop : float, optional
            The (cosmic) time when to derive final relativistic electrons
            spectrum for synchrotron emission calculations.
            Default: ``self.age_obs``.
            Unit: [Gyr]
        n0_e : 1D `~numpy.ndarray`, optional
            The initial electron spectrum (number distribution).
            Default: ``self.electron_spec_init``
            Unit: [cm^-3]

        Returns
        -------
        electron_spec : float 1D `~numpy.ndarray`
            The solved electron spectrum at ``tstop``.
            Unit: [cm^-3]
        """
        if tstart is None:
            tstart = self.age_begin
        if tstop is None:
            tstop = self.age_obs
        if n0_e is None:
            n0_e = self.electron_spec_init

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
        process, it has only an effective acceleration time ~<1 Gyr.
        Therefore, only during the period that strong turbulence persists
        in the ICM that the turbulence could effectively accelerate the
        relativistic electrons.

        WARNING
        -------
        A zero diffusion coefficient may lead to unstable/wrong results,
        since it is not properly taken care of by the solver.  Therefore
        give the acceleration timescale a large enough but finite value
        after turbulent acceleration.
        Also note that a very large acceleration timescale (e.g., 1000 or
        even 10000) will also cause problems (maybe due to some limitations
        within the current calculation scheme), for example, the energy
        losses don't seem to have effect in such cases, so the derived
        initial electron spectrum is almost the same as the raw input one,
        and the emissivity at medium/high frequencies even decreases when
        the turbulence acceleration begins!
        By carrying out some tests, the value of 10 [Gyr] is adopted for
        the maximum acceleration timescale.

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
        # Maximum acceleration timescale when no turbulence acceleration
        # NOTE: see the above WARNING!
        tau_max = 10.0  # [Gyr]
        if (t < self.age_begin) or (t > self.age_begin+self.time_turbulence):
            # NO active turbulence acceleration
            tau_acc = tau_max
        else:
            # Turbulence acceleration
            tau_acc = self.tau_acceleration  # [Gyr]
        # Impose the maximum acceleration timescale
        if tau_acc > tau_max:
            tau_acc = tau_max

        gamma = np.asarray(gamma)
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
        if t < self.age_begin:
            # To derive the initial electron spectrum
            advection = (abs(self._loss_ion(gamma, self.age_begin)) +
                         abs(self._loss_rad(gamma, self.age_begin)))
        else:
            # Turbulence acceleration and beyond
            advection = (abs(self._loss_ion(gamma, t)) +
                         abs(self._loss_rad(gamma, t)) -
                         (self.fp_diffusion(gamma, t) * 2 / gamma))
        return advection

    def _mass(self, t):
        """
        Calculate the main cluster mass at the given (cosmic) time.

        NOTE
        ----
        Since we currently only consider the last major merger event,
        there may be a long time between ``z_merger`` and ``z_obs``.
        So we assume that the main cluster grows linearly in time from
        (M_main, z_merger) to (M_obs, z_obs).

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
        t0 = self.age_begin
        rate = (self.M_obs - self.M_main) / (self.age_obs - t0)
        mass = rate * (t - t0) + self.M_main
        return mass

    def _velocity_turb(self, t=None):
        """
        Calculate the turbulence velocity dispersion (i.e., turbulence
        Mach number).

        NOTE
        ----
        During the merger, a fraction of the merger kinetic energy is
        transferred into the turbulence within the assumed regions
        (radius <= L, the injection scale).  Then estimate the turbulence
        velocity dispersion from its energy.

        Merger energy:
            E_m ≅ 0.5 * f_gas * M_sub * v_vir^2
            v_vir = sqrt(G*M_main / R_vir)
        Turbulence energy:
            E_turb ≅ η_turb * E_m
                   ≅ 0.5 * M_turb * <v_turb^2>
                   = 0.5 * f_gas * M_total(<L) * <v_turb^2>
                   = 0.5 * f_gas * f_mass(L/R_vir) * M_total * <v_turb^2>
            M_total = M_main + M_sub
        => Velocity dispersion:
            <v_turb^2> ≅ (η_turb/f_mass) * (M_sub/M_total) * v_vir^2

        Returns
        -------
        v_turb : float
            The turbulence velocity dispersion
            Unit: [km/s]
        """
        if t is None:
            t = self.age_begin
        z = COSMO.redshift(t)
        mass = self.M_main + self.M_sub
        R_vir = helper.radius_virial(mass=mass, z=z) * AUC.kpc2cm  # [cm]
        v2_vir = (AC.G * self.M_main*AUC.Msun2g / R_vir) * AUC.cm2km**2
        fmass = helper.fmass_nfw(self.f_lturb)
        v2_turb = v2_vir * (self.eta_turb / fmass) * (self.M_sub / mass)
        return np.sqrt(v2_turb)

    def _magnetic_field(self, t):
        """
        Calculate the mean magnetic field strength of the main cluster mass
        at the given (cosmic) time.

        Parameters
        ----------
        t : float
            The (cosmic) time/age.
            Unit: [Gyr]

        Returns
        -------
        B : float
            The mean magnetic field strength of the main cluster.
            Unit: [uG]
        """
        z = COSMO.redshift(t)
        mass = self._mass(t)  # [Msun]
        B = helper.magnetic_field(mass=mass, z=z, configs=self.configs)
        return B

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
        gamma = np.asarray(gamma)
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
        gamma = np.asarray(gamma)
        B = self._magnetic_field(t)  # [uG]
        z = COSMO.redshift(t)
        loss = -4.32e-4 * gamma**2 * ((B/3.25)**2 + (1+z)**4)
        return loss
