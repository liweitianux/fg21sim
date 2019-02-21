# Copyright (c) 2017-2019 Weitian LI <wt@liwt.net>
# MIT License

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

.. [cassano2007]
   Cassano et al. 2007, MNRAS, 378, 1565
   http://adsabs.harvard.edu/abs/2007MNRAS.378.1565C

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

.. [vazza2011]
   Vazza et al. 2011, A&A, 529, A17
   http://adsabs.harvard.edu/abs/2011A%26A...529A..17V
"""

import logging
from functools import lru_cache

import numpy as np
from scipy import integrate

from . import helper
from .solver import FokkerPlanckSolver
from .emission import HaloEmission
from ...share import CONFIGS, COSMO
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)


logger = logging.getLogger(__name__)


class RadioHalo1M:
    """
    Simulate the radio halo properties for a galaxy cluster that is
    experiencing an on-going merger or had a merger recently.

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
    _merger_disabled : bool
        Whether the *merger-induced* turbulence acceleration is
        intentionally disabled (in order to derive the fiducial results)?
    """
    compID = "extragalactic/halos"
    name = "giant radio halos"

    def __init__(self, M_obs, z_obs, M_main, M_sub, z_merger,
                 configs=CONFIGS):
        self.M_obs = M_obs
        self.z_obs = z_obs
        self.t_obs = COSMO.age(z_obs)
        self.M_main = M_main
        self.M_sub = M_sub
        self.z_merger = z_merger
        self.t_merger = COSMO.age(z_merger)

        self._merger_disabled = False
        self._set_configs(configs)
        self._set_solver()

    def _set_configs(self, configs):
        sec = self.compID
        self.configs = configs
        self.f_acc = configs.getn(sec+"/f_acc")
        self.f_radius = configs.getn(sec+"/f_radius")
        self.x_turb = configs.getn(sec+"/x_turb")
        self.eta_turb = configs.getn(sec+"/eta_turb")
        self.eta_e = configs.getn(sec+"/eta_e")
        self.x_cr = configs.getn(sec+"/x_cr")
        self.eta_b = self.x_cr  # Equipartition between magnetic field and CR
        self.mass_index = configs.getn(sec+"/mass_index")
        self.radius_index = configs.getn(sec+"/radius_index")
        self.gamma_min = configs.getn(sec+"/gamma_min")
        self.gamma_max = configs.getn(sec+"/gamma_max")
        self.gamma_np = configs.getn(sec+"/gamma_np")
        self.buffer_np = configs.getn(sec+"/buffer_np")
        if self.buffer_np == 0:
            self.buffer_np = None
        self.time_step = configs.getn(sec+"/time_step")
        self.time_init = configs.getn(sec+"/time_init")
        self.injection_index = configs.getn(sec+"/injection_index")
        self.f_rc = configs.getn(sec+"/f_rc")
        self.beta = configs.getn(sec+"/beta")
        self.bolo_freq_min = configs.getn(sec+"/bolo_freq_min")
        self.bolo_freq_max = configs.getn(sec+"/bolo_freq_max")
        self.bolo_freq_num = configs.getn(sec+"/bolo_freq_num")

        self.kT_out = configs.getn("extragalactic/clusters/kT_out")

    def _set_solver(self):
        self.fpsolver = FokkerPlanckSolver(
            xmin=self.gamma_min,
            xmax=self.gamma_max,
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
    def t_begin(self):
        """
        The cosmic time when the merger begins.
        Unit: [Gyr]
        """
        return self.t_merger

    @property
    @lru_cache()
    def t_merger_end(self):
        """
        The time when the merger (i.e., turbulence) ends.
        Unit: [Gyr]
        """
        return self.t_merger + self.duration_turb(self.t_merger)

    @property
    def radius(self):
        """
        The estimated radius of the simulated radio halo.

        It is known that the halo radius scales non-linearly as the hosting
        cluster, breaking the self-similiarity, which may be caused by the
        magnetic field and the releativistic electron distributions.

        Reference: Ref.[cassano2007],Sec.4

        Unit: [kpc]
        """
        r_turb = self.radius_turb(self.t_merger)
        r_cl = helper.radius_cluster(mass=self.M_obs, z=self.z_obs)
        r0_cl = helper.radius_cluster(mass=1e15, z=0)
        return r_turb * self.f_radius * (r_cl/r0_cl)**self.radius_index

    @lru_cache()
    def radius_strip(self, t_merger):
        """
        The stripping radius of the in-falling sub-cluster at time t.
        Unit: [kpc]
        """
        self._validate_time(t_merger)
        z = COSMO.redshift(t_merger)
        M_main = self.mass_main(t_merger)
        M_sub = self.mass_sub(t_merger)
        return helper.radius_stripping(M_main, M_sub, z,
                                       f_rc=self.f_rc, beta=self.beta)

    @lru_cache()
    def radius_turb(self, t_merger):
        """
        The radius of the turbulence region, which is estimated as the
        sum of stripping radius ``r_s`` of the sub-cluster and the core
        radius ``r_c`` of the main cluster.

        Unit: [kpc]
        """
        self._validate_time(t_merger)
        z = COSMO.redshift(t_merger)
        M_main = self.mass_main(t_merger)
        r_s = self.radius_strip(t_merger)
        r_c = self.f_rc * helper.radius_cluster(M_main, z)
        return r_s + r_c

    @lru_cache()
    def duration_turb(self, t_merger):
        """
        The duration that the turbulence persists strong enough to be able
        to effectively accelerate the electrons, which is estimated as:
            τ_turb ~ d / v_i ~ 2*R_turb / v_i,
        where ``v_i`` is the relative impact velocity.

        Reference: [miniati2015],Sec.5

        Unit: [Gyr]
        """
        self._validate_time(t_merger, include_end=False)
        z_merger = COSMO.redshift(t_merger)
        M_main = self.mass_main(t_merger)
        M_sub = self.mass_sub(t_merger)
        d = 2 * self.radius_turb(t_merger)
        v_i = helper.velocity_impact(M_main, M_sub, z_merger)
        uconv = AUC.kpc2km * AUC.s2Gyr  # [kpc]/[km/s] => [Gyr]
        return uconv * d / v_i  # [Gyr]

    @lru_cache()
    def velocity_turb_base(self, t):
        """
        Calculate the velocity dispersion of the base turbulence.

        Without injection by mergers, the ICM can has some turbulence,
        which can amount ~< 5% of the thermal energy in relaxed systems
        (Ref.[vazza2011]).

        ε_turb = (1/2) * ρ_gas * <v_turb^2> = x_turb * ε_th
        ε_th = (3/2) * n_th * kT
        ρ_gas = μ * m_u * n_th
        c_s^2 = γ_gas * kT / (μ * m_u)
        =>
        v_turb = c_s * sqrt(3 * x_turb / γ_gas)

        Parameters
        ----------
        t : float
            The beginning or ending time of the merger.
            Unit: [Gyr]

        Returns
        -------
        v_turb : float
            The velocity dispersion of the base turbulence.
            Unit: [km/s]
        """
        self._validate_time(t)

        if np.isclose(self.x_turb, 0):
            return 0

        c_s = helper.speed_sound(self.kT(t))  # [km/s]
        return c_s * np.sqrt(3*self.x_turb / AC.gamma)

    @lru_cache()
    def velocity_turb(self, t):
        """
        Calculate the turbulence velocity dispersion.

        NOTE
        ----
        During the merger, a fraction of the merger kinetic energy is
        transferred into the turbulence within the region of radius R_turb.
        Then estimate the turbulence velocity dispersion from its energy.

        Merger energy:
            E_merger ≅ <ρ_gas> * v_i^2 * V_turb
            V_turb = ᴨ * r_s^2 * (R_vir+r_s)
        Turbulence energy:
            E_turb ≅ 0.5 * M_turb * <v_turb^2>
                   ≅ 0.5 * M_turb * <v_turb_base^2> + η_turb * E_merger
        => Velocity dispersion:
            <v_turb^2> ≅ <v_turb_base^2> +
                         2*η_turb * <ρ_gas> * v_i^2 * V_turb / M_turb
            M_turb = int_0^R_turb[ ρ_gas(r)*4ᴨ*r^2 ]dr
        where:
            <ρ_gas>: mean gas density of the main cluster
            R_vir: virial radius of the main cluster
            R_turb: radius of turbulence region
            v_i: impact velocity
            r_s: stripping radius of the in-falling sub-cluster

        Returns
        -------
        v_turb : float
            The turbulence velocity dispersion
            Unit: [km/s]
        """
        self._validate_time(t)
        z = COSMO.redshift(t)
        M_main = self.mass_main(t)
        M_sub = self.mass_sub(t)
        r_s = self.radius_strip(t)  # [kpc]
        R_turb = self.radius_turb(t)  # [kpc]
        v_turb_base = self.velocity_turb_base(t)  # [km/s]

        rho_gas_f = helper.calc_gas_density_profile(
                M_main+M_sub, z, f_rc=self.f_rc, beta=self.beta)
        M_turb = 4*np.pi * integrate.quad(
                lambda r: rho_gas_f(r) * r**2,
                a=0, b=R_turb)[0]  # [Msun]

        v_i = helper.velocity_impact(M_main, M_sub, z)  # [km/s]
        rho_main = helper.density_number_thermal(M_main, z)  # [cm^-3]
        rho_main *= AC.mu*AC.u * AUC.g2Msun * AUC.kpc2cm**3  # [Msun/kpc^3]
        R_vir = helper.radius_cluster(M_main, z)  # [kpc]

        volume = np.pi * r_s**2 * R_vir  # [kpc^3]
        E_add = self.eta_turb * rho_main * v_i**2 * volume
        v2_turb = v_turb_base**2 + 2 * E_add / M_turb  # [km/s]^2
        return np.sqrt(v2_turb)

    @lru_cache()
    def tau_acceleration(self, t, has_merger=True):
        """
        Calculate the electron acceleration timescale due to turbulent
        waves, which describes the turbulent acceleration efficiency.

        Here we consider the turbulence cascade mode through scattering
        in the high-β ICM mediated by plasma instabilities (firehose,
        mirror) rather than Coulomb scattering.  Therefore, the fast modes
        damp by TTD (transit time damping) on relativistic rather than
        thermal particles, and the diffusion coefficient is given by:
            D'_γγ = 2 * γ^2 * ζ * k_L * v_t^4 / (c_s^3 * X_cr)
        where:
            ζ: factor describing the effectiveness of plasma instabilities
            X_cr: relative energy density of cosmic rays
            k_L (= 2π/L): turbulence injection scale
            v_t: turbulence velocity dispersion
            c_s: sound speed
        Hence, the acceleration timescale is:
            τ'_acc = γ^2 / (4 * D'_γγ)
                   = X_cr * c_s^3 / (8 * ζ * k_L * v_t^4)

        Previous studies show that more massive clusters are more efficient
        to accelerate electrons to be radio bright.  To further account for
        this scaling relation:
            D_γγ = D'_γγ * f_m * (M_main / 1e15)^m
        where:
            m: scaling index
            f_m: normalization
        Therefore, the final acceleration timescale is:
            τ_acc = τ'_acc * (M_main / 1e15)^(-m) / f_m
                  = X_cr * c_s^3 * (M_main/1e15)^(-m) / (8*f_acc * k_L * v_t^4)
        with: f_acc = f_m * ζ

        References
        ----------
        * Ref.[pinzke2017],Eq.(37)
        * Ref.[miniati2015],Eq.(29)

        Parameters
        ----------
        t : float
            The beginning or ending time of the merger.
            Unit: [Gyr]
        has_merger : bool, optional
            If ``False``, calculate the velocity dispersion of the base
            turbulence, i.e., without a merger.
            Default: True

        Returns
        -------
        tau : float
            The acceleration timescale of the merger.
            Unit: [Gyr]
        """
        self._validate_time(t)

        if has_merger:
            v_t = self.velocity_turb(t)  # [km/s]
        else:
            v_t = self.velocity_turb_base(t)  # [km/s]

        if np.isclose(v_t, 0):
            return np.inf

        k_L = 2 * np.pi / self.radius_turb(t)  # [kpc^-1]
        c_s = helper.speed_sound(self.kT(t))  # [km/s]
        tau = self.x_cr * c_s**3 / (8*k_L * v_t**4)
        tau *= AUC.s2Gyr * AUC.kpc2km  # [s kpc/km] -> [Gyr]

        # Scale by cluster mass
        M_main = self.mass_main(t)
        f_mass = (M_main / 1e15) ** (-self.mass_index)
        tau *= f_mass
        tau /= self.f_acc  # tune factor (folded with ζ "zeta_ins")

        return tau

    @property
    @lru_cache()
    def injection_rate(self):
        """
        The constant electron injection rate assumed.
        Unit: [cm^-3 Gyr^-1]

        The injection rate is parametrized by assuming that the total
        energy injected in the relativistic electrons during the cluster
        life (e.g., ``t_obs`` here) is a fraction (``self.eta_e``)
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
        kT_out = self.configs.getn("extragalactic/clusters/kT_out")
        s = self.injection_index
        e_th = helper.density_energy_thermal(self.M_obs, self.z_obs,
                                             kT_out=kT_out)
        term1 = (s-2) * self.eta_e * e_th  # [erg cm^-3]
        term2 = self.gamma_min**(s-2)
        term3 = AU.mec2 * self.t_obs  # [erg Gyr]
        Ke = term1 * term2 / term3  # [cm^-3 Gyr^-1]
        return Ke

    @property
    def electron_spec_init(self):
        """
        The electron spectrum at ``t_begin`` to be used as the initial
        condition for the Fokker-Planck equation.

        This initial electron spectrum is derived from the accumulated
        electron spectrum injected throughout the ``t_begin`` period,
        by solving the same Fokker-Planck equation, but only considering
        energy losses and constant injection, evolving for a period of
        ``time_init`` in order to obtain an approximately steady electron
        spectrum.

        Units: [cm^-3]
        """
        # Accumulated electrons constantly injected until ``t_begin``
        n_inj = self.fp_injection(self.gamma)
        n0_e = n_inj * (self.t_begin - self.time_init)

        logger.debug("Deriving the initial electron spectrum ...")
        self._merger_disabled = True
        tstart = self.t_begin
        tstop = self.t_begin + self.time_init
        self.fpsolver.tstep = self.time_step * 3  # To save time

        n_e = self.fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
        self._merger_disabled = False
        self.fpsolver.tstep = self.time_step

        return n_e

    def calc_electron_spectrum(self, tstart=None, tstop=None, n0_e=None,
                               fiducial=False):
        """
        Calculate the relativistic electron spectrum by solving the
        Fokker-Planck equation.

        Parameters
        ----------
        tstart : float, optional
            The (cosmic) time from when to solve the Fokker-Planck equation
            for relativistic electrons evolution.
            Default: ``self.t_begin``.
            Unit: [Gyr]
        tstop : float, optional
            The (cosmic) time when to derive final relativistic electrons
            spectrum for synchrotron emission calculations.
            Default: ``self.t_obs``.
            Unit: [Gyr]
        n0_e : 1D `~numpy.ndarray`, optional
            The initial electron spectrum (number distribution).
            Default: ``self.electron_spec_init``
            Unit: [cm^-3]
        fiducial : bool
            Whether to disable the merger-induced turbulent acceleration
            and derive the fiducial electron spectrum?
            Default: ``False``

        Returns
        -------
        n_e : float 1D `~numpy.ndarray`
            The solved electron spectrum.
            Unit: [cm^-3]
        """
        if tstart is None:
            tstart = self.t_begin
        if tstop is None:
            tstop = self.t_obs
        if n0_e is None:
            n0_e = self.electron_spec_init
        if fiducial:
            self._merger_disabled = True
            self.fpsolver.tstep = self.time_step * 2  # To save time

        logger.debug("Calculating the %s electron spectrum ..." %
                     ("[fiducial]" if fiducial else ""))
        n_e = self.fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
        self._merger_disabled = False
        self.fpsolver.tstep = self.time_step

        return n_e

    def calc_acc_factor(self, n_e, n_e_fiducial=None):
        """
        Calculate the turbulence acceleration factor, which is estimated
        as the ratio of the bolometric emissivity between the accelerated
        electron spectrum and the fiducial electron spectrum derived with
        turbulent acceleration turned off.

        Parameters
        ----------
        n_e : float 1D `~numpy.ndarray`
            The derived (accelerated) electron spectrum.
            Unit: [cm^-3]
        n_e_fiducial : float 1D `~numpy.ndarray`, optional
            The fiducial electron spectrum.
            Unit: [cm^-3]

        Returns
        -------
        factor : float
            Acceleration factor of the bolometric emissivity.
        """
        haloem = HaloEmission(gamma=self.gamma, n_e=n_e, B=1)
        em = haloem.calc_emissivity_bolo(freq_min=self.bolo_freq_min,
                                         freq_max=self.bolo_freq_max,
                                         freq_num=self.bolo_freq_num)

        if n_e_fiducial is None:
            n_e_fiducial = self.calc_electron_spectrum(fiducial=True)
        haloem.n_e = n_e_fiducial
        em_fiducial = haloem.calc_emissivity_bolo(
                freq_min=self.bolo_freq_min, freq_max=self.bolo_freq_max,
                freq_num=self.bolo_freq_num)

        return em / em_fiducial

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
        and calculated from the acceleration timescale ``tau_acc``.

        WARNING
        -------
        A zero diffusion coefficient may lead to unstable/wrong results,
        since it is not properly taken care of by the solver.
        By carrying out some tests, the maximum acceleration timescale
        ``tau_acc`` is assumed to be 10 [Gyr].

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
        """
        tau_max = 10.0  # [Gyr]

        if self._is_turb_active(t):
            tt = self._merger_time(t)
            has_merger = True
        else:
            tt = self._merger_end_time(t)
            has_merger = False
        if self._merger_disabled:
            has_merger = False

        tau = self.tau_acceleration(tt, has_merger=has_merger)
        if tau > tau_max:
            tau = tau_max

        return np.square(gamma) / (4 * tau)  # [Gyr^-1]

    def fp_advection(self, gamma, t):
        """
        Advection term/coefficient for the Fokker-Planck equation,
        which describes a systematic tendency for upward or downward
        drift of particles.

        This term is also called the "generalized cooling function"
        by [donnert2014], which includes all relevant energy loss
        functions and the energy gain function due to turbulence.

        Returns
        -------
        advection : float, or float 1D `~numpy.ndarray`
            Advection coefficient.
            Unit: [Gyr^-1]
        """
        advection = (abs(self._energy_loss(gamma, t)) -
                     (self.fp_diffusion(gamma, t) * 2 / gamma))
        return advection

    def _energy_loss(self, gamma, t):
        """
        Energy loss mechanisms:
        * inverse Compton scattering off the CMB photons
        * synchrotron radiation
        * Coulomb collisions

        Reference: Ref.[sarazin1999],Eqs.(6,7,9)

        Unit: [Gyr^-1]
        """
        gamma = np.asarray(gamma)
        z = COSMO.redshift(t)
        B = self.magnetic_field(t)  # [uG]
        mass = self.mass_main(t)
        n_th = helper.density_number_thermal(mass, z)  # [cm^-3]
        loss_ic = -4.32e-4 * gamma**2 * (1+z)**4
        loss_syn = -4.10e-5 * gamma**2 * B**2
        loss_coul = -3.79e4 * n_th * (1 + np.log(gamma/n_th) / 75)
        return loss_ic + loss_syn + loss_coul

    def _merger_time(self, t=None):
        """
        The (cosmic) time when the merger begins.
        Unit: [Gyr]
        """
        return self.t_merger

    def _merger_end_time(self, t=None):
        """
        The (cosmic) time when the merger ends.
        Unit: [Gyr]
        """
        return self.t_merger_end

    def _validate_time(self, t, include_end=True):
        """
        Validate that the given time ``t`` is the time when a merger begins
        or ends (if ``include_end`` is ``True``), otherwise raise an error.
        """
        if np.any(np.isclose(t, self.t_merger)):
            return
        if include_end and np.any(np.isclose(t, self.t_merger_end)):
            return
        raise ValueError("Not a merger begin/end time: %f" % t)

    def _is_turb_active(self, t):
        """
        Is the turbulence acceleration active at the given time?
        """
        t_begin = self._merger_time(t)
        t_end = t_begin + self.duration_turb(t_begin)
        return (t >= t_begin) and (t <= t_end)

    def mass_merged(self, t=None):
        """
        The mass of the merged cluster.
        Unit: [Msun]
        """
        return self.M_main + self.M_sub

    def mass_sub(self, t=None):
        """
        The mass of the sub cluster.
        Unit: [Msun]
        """
        return self.M_sub

    def mass_main(self, t):
        """
        Calculate the main cluster mass at the given (cosmic) time.
        The main cluster is assumed to grow linearly in time from
        (M_main, z_merger) to (M_obs, z_obs).

        Unit: [Msun]
        """
        t0 = self.t_begin
        rate = (self.M_obs - self.M_main) / (self.t_obs - t0)
        mass = rate * (t - t0) + self.M_main  # [Msun]
        return mass

    def kT(self, t):
        """
        The ICM mean temperature of the main cluster.
        Unit: [keV]
        """
        kT_out = self.configs.getn("extragalactic/clusters/kT_out")
        M_main = self.mass_main(t)
        z = COSMO.redshift(t)
        return helper.kT_cluster(mass=M_main, z=z, kT_out=kT_out)

    def magnetic_field(self, t):
        """
        Calculate the mean magnetic field strength of the main cluster mass
        at the given (cosmic) time.

        Unit: [uG]
        """
        z = COSMO.redshift(t)
        mass = self.mass_main(t)  # [Msun]
        return helper.magnetic_field(mass=mass, z=z,
                                     eta_b=self.eta_b, kT_out=self.kT_out)


class RadioHaloAM(RadioHalo1M):
    """
    Simulate the radio halo properties for a galaxy cluster with all its
    on-going merger and past merger events taken into account.

    Parameters
    ----------
    M_obs : float
        Cluster virial mass at the observation (simulation end) time.
        Unit: [Msun]
    z_obs : float
        Redshift of the observation (simulation end) time.
    M_main, M_sub : list[float]
        List of main and sub cluster masses at each merger event,
        from current to earlier time.
        Unit: [Msun]
    z_merger : list[float]
        The redshifts at each merger event, from small to large.
    merger_num : int
        Number of merger events traced for the cluster.
    """
    def __init__(self, M_obs, z_obs, M_main, M_sub, z_merger,
                 merger_num, configs=CONFIGS):
        M_main = np.asarray(M_main[:merger_num])
        M_sub = np.asarray(M_sub[:merger_num])
        z_merger = np.asarray(z_merger[:merger_num])  # increasing
        super().__init__(M_obs=M_obs, z_obs=z_obs,
                         M_main=M_main, M_sub=M_sub,
                         z_merger=z_merger, configs=configs)
        self.merger_num = merger_num

    @property
    def t_begin(self):
        """
        The cosmic time when the first merger begins.
        Unit: [Gyr]
        """
        return self.t_merger[-1]

    @property
    @lru_cache()
    def t_merger_end(self):
        """
        The times when the mergers (i.e., turbulence) end.
        (NOTE: decreasing order)

        Unit: [Gyr]
        """
        return np.array([t+self.duration_turb(t) for t in self.t_merger])

    @property
    def radius(self):
        """
        The halo radius estimated by using the maximum turbulence radius.
        Unit: [kpc]
        """
        r_turb = self.radius_turb_max
        r_cl = helper.radius_cluster(mass=self.M_obs, z=self.z_obs)
        r0_cl = helper.radius_cluster(mass=1e15, z=0)
        return r_turb * self.f_radius * (r_cl/r0_cl)**self.radius_index

    @property
    @lru_cache()
    def radius_turb_max(self):
        """
        The maximum turbulence radius.
        Unit: [kpc]
        """
        return max([self.radius_turb(tm) for tm in self.t_merger])

    def radius_turb_eff(self, t, use_last=True):
        """
        Get the effective turbulence radius, i.e., the largest one if
        multiple mergers are active at the given time.

        Parameters
        ----------
        use_last : bool
            If ``True``, return the turbulence radius of the last merger
            event when there is no active turbulence at the given time.
            Otherwise, return 0.

        Unit: [kpc]
        """
        mergers = [(t, t+self.duration_turb(t), self.radius_turb(t))
                   for t in self.t_merger]  # time decreasing
        try:
            r_eff = max([r for t1, t2, r in mergers if t >= t1 and t < t2])
        except ValueError:
            # No active turbulence at this time
            if use_last:
                r_eff = next(r for __, t2, r in mergers if t >= t2)
            else:
                r_eff = 0

        return r_eff

    def _merger_event(self, t):
        """
        Return the most recent merger event happend before the given time,
        i.e., the merger event that the given time locates in.
        """
        idx = (self.t_merger > t).sum()  # 't_merger' in decreasing order
        return {
            "idx": idx,
            "M_main": self.M_main[idx],
            "M_sub": self.M_sub[idx],
            "z": self.z_merger[idx],
            "t": self.t_merger[idx],
        }

    def mass_merged(self, t):
        """
        The mass of merged cluster at the given (cosmic) time.
        Unit: [Msun]
        """
        if t >= self.t_obs:
            return self.M_obs
        else:
            merger = self._merger_event(t)
            return (merger["M_main"] + merger["M_sub"])

    def mass_sub(self, t):
        """
        The mass of the sub cluster at the given (cosmic) time.
        Unit: [Msun]
        """
        merger = self._merger_event(t)
        return merger["M_sub"]

    def mass_main(self, t):
        """
        Calculate the main cluster mass, which is assumed to grow along
        the merger/accretion processes, at the given (cosmic) time.

        Unit: [Msun]
        """
        merger1 = self._merger_event(t)
        idx1 = merger1["idx"]
        mass1 = merger1["M_main"]
        t1 = merger1["t"]
        if idx1 == 0:
            mass0 = self.M_obs
            t0 = self.t_obs
        else:
            idx0 = idx1 - 1
            mass0 = self.M_main[idx0]
            t0 = self.t_merger[idx0]
        rate = (mass0 - mass1) / (t0 - t1)
        return (mass1 + rate * (t - t1))

    def _merger_time(self, t):
        """
        Determine the beginning time of the merger event that is doing
        effective acceleration at the given time.

        At a certain time, there may be multiple past merger events with
        different turbulence durations (``tau_turb``) and acceleration
        efficiencies (``tau_acc``).  Therefore, multiple mergers can cover
        the given time.  The one with the largest acceleration efficiency
        (i.e., smallest ``tau_acc``) is chosen and its beginning time is
        returned.  Otherwise, the most recent merger event happened before
        the given time is chosen.

        Unit: [Gyr]
        """
        mergers = [(tm, tm+self.duration_turb(tm), self.tau_acceleration(tm))
                   for tm in self.t_merger]
        m_active = [(tm, tend, tau) for (tm, tend, tau) in mergers
                    if t >= tm and t < tend]
        if m_active:
            m_eff = min(m_active, key=lambda item: item[2])
            return m_eff[0]
        else:
            m = self._merger_event(t)
            return m["t"]

    def _merger_end_time(self, t):
        """
        Determine the ending time of the most recent merger happened before
        the given time.

        Unit: [Gyr]
        """
        t_end = np.sort(self.t_merger_end)  # increasing order
        idx = (t_end <= t).sum() - 1
        return t_end[idx]

    @property
    @lru_cache()
    def _time_adjust(self):
        """
        Determine the time points when spectrum adjustment is needed.

        Different mergers generate turbulence in regions of different radius,
        therefore, the accelerated spectrum needs appropriate adjustment.

        Returns
        -------
        t_adj : list[float]
            List of (cosmic) times when the adjustment is needed.
            NOTE: May be empty, e.g., only one merger event.
            Unit: [Gyr]
        """
        mergers = [(t, t+self.duration_turb(t)) for t in self.t_merger]
        t_begin = [t for t, __ in mergers]
        t_end = [t for __, t in mergers]
        tps = sorted(t_begin + t_end)
        radii = [self.radius_turb_eff(t, use_last=False) for t in tps]
        tinfo = {t: {"begin": t in t_begin, "end": t in t_end, "radius": r}
                 for t, r in zip(tps, radii)}

        t_adj = []
        for r1, r2, t in zip(radii, radii[1:], tps[1:]):
            if np.isclose(r1, r2):
                continue
            ti = tinfo[t]
            if ti["end"] and np.isclose(ti["radius"], 0):
                continue
            t_adj.append(t)

        return t_adj

    def _adjust_spectrum(self, spec_in, t, spec_ref):
        """
        Adjust the electron spectrum to take into account the change of
        turbulence region size.  If the current turbulence radius is
        smaller than the maximum turbulence radius, then dilute the
        accelerated part of the spectrum according to the volume ratio.

        Parameters
        ----------
        spec_in : 1D `~numpy.ndarray`
            The spectrum at the ending of the given acceleration period.
        t : float
            The corresponding time of the given spectrum ``spec_in``.
            Unit: [Gyr]
        spec_ref : 1D `~numpy.ndarray`
            The spectrum at the beginning of the given acceleration period.

        Returns
        -------
        spec : Adjusted spectrum.
        """
        r = self.radius_turb_eff(t, use_last=True)
        r_max = self.radius_turb_max
        if np.isclose(r, r_max):
            return spec_in

        logger.debug("Adjusting the accelerated spectrum ...")
        spec_diff = spec_in - spec_ref
        idx = spec_diff > 0
        spec = np.array(spec_ref)
        spec[idx] += spec_diff[idx] * (r/r_max)**3
        return spec

    def calc_electron_spectrum(self, tstart=None, tstop=None, n0_e=None,
                               fiducial=False):
        """
        Calculate the relativistic electron spectrum by solving the
        Fokker-Planck equation.

        Given that different mergers have different turbulence radii, the
        spectrum needs appropriate adjustments to take this into account.
        At the beginning of each merger, the accelerated part of the
        spectrum (i.e., where the electron density increases compared to
        last adjustment) is scaled according to the ratio of the previous
        turbulence volume to the maximum turbulence volume, i.e., dilute
        the accelerated spectrum to the maximum turbulence volume.
        """
        if tstart is None:
            tstart = self.t_begin
        if tstop is None:
            tstop = self.t_obs
        if n0_e is None:
            n0_e = self.electron_spec_init

        if fiducial:
            self._merger_disabled = True
            self.fpsolver.tstep = self.time_step * 2  # To save time
            logger.debug("Calculating the [fiducial] electron spectrum ...")
            n_e = self.fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
            self._merger_disabled = False
            self.fpsolver.tstep = self.time_step
            return n_e

        logger.debug("Calculating the electron spectrum ...")
        tps = [self.t_begin] + self._time_adjust + [self.t_obs]
        n1_e = n0_e
        for t1, t2 in zip(tps, tps[1:]):
            if tstart >= t2 or tstop < t1:
                continue
            if tstart > t1:
                t1 = tstart
            if tstop < t2:
                t2 = tstop
            logger.debug("Time period: [%.2f, %.2f] [Gyr] ..." % (t1, t2))
            n2_e = self.fpsolver.solve(u0=n1_e, tstart=t1, tstop=t2)
            n2_e = self._adjust_spectrum(n2_e, t2, spec_ref=n1_e)
            n1_e = n2_e

        return n2_e
