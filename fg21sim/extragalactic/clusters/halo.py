# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Simulate (giant) radio halos following the "statistical
magneto-turbulent model" proposed by Cassano & Brunetti (2005).

References
----------
[1] Cassano & Brunetti 2005, MNRAS, 357, 1313
    http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
[2] Cassano, Brunetti & Setti, 2006, MNRAS, 369, 1577
    http://adsabs.harvard.edu/abs/2006MNRAS.369.1577C
[3] Cassano et al. 2012, A&A, 548, A100
    http://adsabs.harvard.edu/abs/2012A%26A...548A.100C
"""

import logging

import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.optimize

from .formation import ClusterFormation
from .solver import FokkerPlanckSolver
from ...utils.cosmology import Cosmology
from ...utils.units import (Units as AU,
                            UnitConversions as AUC,
                            Constants as AC)


logger = logging.getLogger(__name__)


class HaloSingle:
    """
    Simulate a single (giant) radio halos following the "statistical
    magneto-turbulent model" proposed by Cassano & Brunetti (2005).

    First, simulate the cluster merging history from the extended
    Press-Schecter formalism using the Monte Carlo method; then derive
    the merger energy and turbulence energy as well as its spectrum;
    after that, calculate the electron acceleration and time  evolution
    by solving the Fokker-Planck equation; and finally derive the radio
    emission from the electron spectra.

    Parameters
    ----------
    M0 : float
        Cluster virial mass at redshift z0
        Unit: [Msun]
    z0 : float
        Redshift from where to simulate former merging history.
    configs : `ConfigManager`
        A `ConfigManager` instance containing default and user configurations.
        For more details, see the example configuration specifications.

    Attributes
    ----------
    mec : float
        Unit for electron momentum (p): mec = m_e * c, p = gamma * mec,
        therefore value of p is the Lorentz factor.
    cosmo : `~Cosmology`
        Adopted cosmological model with custom utility functions.
    mtree : `~MergerTree`
        Merging history of this cluster.
    """
    # Unit for electron momentum (p), thus its value is the Lorentz factor
    mec = AU.mec  # [g cm / s]
    # Merger tree (i.e., merging history) of this cluster
    mtree = None

    def __init__(self, M0, z0, configs):
        self.M0 = M0  # [Msun]
        self.z0 = z0
        self.configs = configs
        self._set_configs()

    def _set_configs(self):
        """
        Set up the necessary class attributes according to the configs.
        """
        comp = "extragalactic/halos"
        self.zmax = self.configs.getn(comp+"/zmax")
        # Mass threshold of the sub-cluster for a significant merger
        self.merger_mass_th = self.configs.getn(comp+"/merger_mass_th")
        self.radius = self.configs.getn(comp+"/radius")
        self.b_mean = self.configs.getn(comp+"/b_mean")
        self.b_index = self.configs.getn(comp+"/b_index")
        self.eta_t = self.configs.getn(comp+"/eta_t")
        self.eta_e = self.configs.getn(comp+"/eta_e")
        self.pmin = self.configs.getn(comp+"/pmin")
        self.pmax = self.configs.getn(comp+"/pmax")
        self.pgrid_num = self.configs.getn(comp+"/pgrid_num")
        self.buffer_np = self.configs.getn(comp+"/buffer_np")
        self.time_step = self.configs.getn(comp+"/time_step")
        self.injection_index = self.configs.getn(comp+"/injection_index")
        # Cosmology model
        self.H0 = self.configs.getn("cosmology/H0")
        self.OmegaM0 = self.configs.getn("cosmology/OmegaM0")
        self.cosmo = Cosmology(H0=self.H0, Om0=self.OmegaM0)
        logger.info("Loaded and set up configurations")

    @property
    def magnetic_field(self):
        """
        Calculate the mean magnetic field of this cluster from the
        scaling relation between magnetic field and cluster mass.

        Unit: [uG]

        Reference: Ref.[3],Eq.(1)
        """
        M_mean = 1.6e15  # [Msun]
        B = self.b_mean * (self.M0/M_mean) ** self.b_index
        return B

    def simulate_mergertree(self):
        """
        Simulate the merging history of the cluster using the extended
        Press-Schechter formalism.

        Attributes
        ----------
        mtree : `~MergerTree`
            Generated merger tree of this cluster.

        Returns
        -------
        mtree : `~MergerTree`
            Generated merger tree of this cluster.
        """
        self.formation = ClusterFormation(self.M0, self.z0, self.configs)
        self.mtree = self.formation.simulate_mergertree()
        return self.mtree

    def calc_electron_spectrum(self, zbegin=None, zend=None):
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
            Default: 0 (i.e., present)

        Returns
        -------
        p : `~numpy.ndarray`
            The momentum grid adopted for solving the equation.
            Unit: [mec]
        n_e : `~numpy.ndarray`
            The solved electron spectrum at ``zend``.
            Unit: [cm^-3 mec^-1]
        """
        fpsolver = FokkerPlanckSolver(
            xmin=self.pmin, xmax=self.pmax,
            grid_num=self.pgrid_num,
            buffer_np=self.buffer_np,
            tstep=self.time_step,
            f_advection=self.fp_advection,
            f_diffusion=self.fp_diffusion,
            f_injection=self.fp_injection,
        )
        p = fpsolver.x
        # Assume NO initial electron distribution
        n0_e = np.zeros(p.shape)
        if zbegin is None:
            tstart = self.cosmo.age(self.zmax)
        else:
            tstart = self.cosmo.age(zbegin)
        if zend is None:
            tstop = self.cosmo.age0
        else:
            tstop = self.cosmo.age(zend)
        n_e = fpsolver.solve(u0=n0_e, tstart=tstart, tstop=tstop)
        return (p, n_e)

    def kT_mass(self, mass):
        """
        Estimate the cluster ICM temperature from its mass by assuming
        an (observed) temperature-mass relation.

        TODO: upgrade this M-T relation.

        Parameters
        ----------
        mass : float
            Mass (unit: Msun) of the cluster

        Returns
        -------
        kT : float
            Temperature of the ICM (unit: keV)

        References
        ----------
        [1] Nevalainen et al. 2000, ApJ, 532, 694
            Ettori et al, 2013, Space Science Review, 177, 119-154
            NOTE: H0 = 50 * h50 [km/s/Mpc]
        """
        kT = 10 * (mass/1.23e15) ** (1/1.79)  # [keV]
        return kT

    def _radius_virial(self, mass, z=0.0):
        """
        Calculate the virial radius of a cluster.

        Parameters
        ----------
        mass : float
            Mass (unit: Msun) of the cluster
        z : float
            Redshift

        Returns
        -------
        Rvir : float
            Virial radius (unit: kpc) of the cluster at given redshift
        """
        Dc = self.cosmo.overdensity_virial(z)
        rho = self.cosmo.rho_crit(z)  # [g/cm^3]
        R_vir = (3*mass*AUC.Msun2g / (4*np.pi * Dc * rho))**(1/3)  # [cm]
        R_vir *= AUC.cm2kpc  # [kpc]
        return R_vir

    def _radius_stripping(self, mass, M_main, z):
        """
        Calculate the stripping radius of the sub-cluster at which
        equipartition between static and ram pressure is established,
        and the stripping is efficient outside this stripping radius.

        Note that the value of the stripping radius obtained would
        give the *mean value* of the actual stripping radius during
        a merger.

        Parameters
        ----------
        mass : float
            The mass (unit: Msun) of the sub-cluster.
        M_main : float
            The mass (unit: Msun) of the main cluster.
        z : float
            Redshift

        Returns
        -------
        rs : float
            The stripping radius of the sub-cluster.
            Unit: kpc

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(11)
        """
        vi = self._velocity_impact(M_main, mass, z) * 1e5  # [cm/s]
        kT = self.kT_mass(mass) * AUC.keV2erg  # [erg]
        coef = kT / (AC.mu * AC.u * vi**2)  # dimensionless
        rho_avg = self._density_average(M_main, z)  # [g/cm^3]

        def equation(r):
            return coef * self.density_profile(r, mass, z) / rho_avg - 1

        r_vir = self._radius_virial(mass, z)  # [kpc]
        rs = scipy.optimize.brentq(equation, a=0, b=r_vir)  # [kpc]
        return rs

    def _density_average(self, mass, z=0.0):
        """
        Average density of the cluster ICM.

        Returns
        -------
        rho : float
            Average ICM density (unit: g/cm^3)

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(12)
        """
        f_baryon = self.cosmo.Ob0 / self.cosmo.Om0
        Rv = self._radius_virial(mass, z) * AUC.kpc2cm  # [cm]
        V = (4*np.pi / 3) * Rv**3  # [cm^3]
        rho = f_baryon * mass*AUC.Msun2g / V  # [g/cm^3]
        return rho

    def density_profile(self, r, mass, z):
        """
        ICM (baryon) density profile, assuming the beta model.

        Parameters
        ----------
        r : float
            Radius (unit: kpc) where to calculate the density
        mass : float
            Cluster mass (unit: Msun)
        z : float
            Redshift

        Returns
        -------
        rho_r : float
            Density at the specified radius (unit: g/cm^3)

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(13)
        """
        f_baryon = self.cosmo.Ob0 / self.cosmo.Om0
        M_ICM = mass * f_baryon * AUC.Msun2g  # [g]
        r *= AUC.kpc2cm  # [cm]
        Rv = self._radius_virial(mass, z) * AUC.kpc2cm  # [cm]
        rc = self._beta_rc(Rv)
        beta = self._beta_beta()
        norm = self._beta_norm(M_ICM, beta, rc, Rv)  # [g/cm^3]
        rho_r = norm * (1 + (r/rc)**2) ** (-3*beta/2)  # [g/cm^3]
        return rho_r

    @staticmethod
    def _beta_rc(r_vir):
        """
        Core radius of the beta model for the ICM density profile.

        TODO: upgrade this!
        """
        return 0.1*r_vir

    @staticmethod
    def _beta_beta():
        """
        Beta value of the beta model for the ICM density profile.

        TODO: upgrade this!
        """
        return 0.8

    @staticmethod
    def _beta_norm(mass, beta, rc, r_vir):
        """
        Calculate the normalization of the beta model for the ICM
        density profile.

        Parameters
        ----------
        mass : float
            The mass (unit: g) of ICM
        beta : float
            Beta value of the assumed beta profile
        rc : float
            Core radius (unit: cm) of the assumed beta profile
        r_vir : float
            The virial radius (unit: cm) of the cluster

        Returns
        -------
        norm : float
            Normalization of the beta model (unit: g/cm^3)

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(14)
        """
        integration = scipy.integrate.quad(
            lambda r: r*r * (1+(r/rc)**2) ** (-3*beta/2),
            0, r_vir)[0]
        norm = mass / (4*np.pi * integration)  # [g/cm^3]
        return norm

    def _velocity_impact(self, M_main, M_sub, z=0.0):
        """
        Calculate the relative impact velocity between the two merging
        clusters when they are at a distance of virial radius.

        Parameters
        ----------
        M_main : float
            Mass of the main cluster (unit: Msun)
        M_sub : float
            Mass of the sub cluster (unit: Msun)
        z : float
            Redshift

        Returns
        -------
        vi : float
            Relative impact velocity (unit: km/s)

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(9)
        """
        eta_v = 4 * (1 + M_main/M_sub) ** (1/3)
        R_vir = self._radius_virial(M_main, z) * AUC.kpc2cm  # [cm]
        vi = np.sqrt(2*AC.G * (1-1/eta_v) *
                     (M_main+M_sub)*AUC.Msun2g / R_vir)  # [cm/s]
        vi /= AUC.km2cm  # [km/s]
        return vi

    def _time_crossing(self, M_main, M_sub, z):
        """
        Calculate the crossing time of the sub-cluster during a merger.

        Parameters
        ----------
        M_main : float
            Mass of the main cluster (unit: Msun)
        M_sub : float
            Mass of the sub cluster (unit: Msun)
        z : float
            Redshift where the merger occurs.

        Returns
        -------
        time : float
            Crossing time (unit: Gyr)
        """
        R_vir = self._radius_virial(M_main, z)  # [kpc]
        vi = self._velocity_impact(M_main, M_sub, z)  # [km/s]
        # Unit conversion coefficient: [s kpc/km] => [Gyr]
        uconv = AUC.kpc2km * AUC.s2Gyr
        time = uconv * R_vir / vi  # [Gyr]
        return time

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
        t_begin = self.cosmo.age(z_begin)  # [Gyr]
        t_end = t_begin + time
        if t_end >= self.cosmo.age(0):
            z_end = 0.0
        else:
            z_end = self.cosmo.redshift(t_end)
        return z_end

    @property
    def merger_events(self):
        """
        Trace only the main cluster, and filter out the significant
        merger events.

        Returns
        -------
        mevents : list[dict]
            List of dictionaries that records all the merger events
            of the main cluster.
            NOTE:
            The merger events are ordered by increasing redshifts.
        """
        events = []
        tree = self.mtree
        while tree:
            if (tree.main and tree.sub and
                    tree.sub.data["mass"] >= self.merger_mass_th and
                    tree.main.data["z"] <= self.zmax):
                events.append({
                    "z": tree.main.data["z"],
                    "age": tree.main.data["age"],
                    "M_main": tree.main.data["mass"],
                    "M_sub": tree.sub.data["mass"],
                })
            tree = tree.main
        return events

    def _coef_acceleration(self, z):
        """
        Calculate the electron-acceleration coefficient at arbitrary
        redshift.

        Parameters
        ----------
        z : float
            Redshift where to calculate the acceleration coefficient.

        Returns
        -------
        chi : float
            The calculated electron-acceleration coefficient.
            (unit: Gyr^-1)

        Attributes
        ----------
        _coef_acceleration_data : (`~numpy.ndarray`, `~numpy.ndarray`)
            (zgrid, chigrid) tuple with ``zgrid`` the array of redshifts
            grid with even spacing ``zbs``, and ``chigrid`` the array
            of acceleration coefficients at each redshift.

        XXX/NOTE
        --------
        This coefficient may be very small and even zero, then the
        diffusion coefficient of the Fokker-Planck equation is thus
        very small and even zero, which cause problems for calculating
        some quantities (e.g., w(x), C(x)) and wrong/invalid results.
        To avoid these problems, force the minimal value of this
        coefficient to be 1/(10*t0), which t0 is the present-day age
        of the universe.
        """
        # Redshift bin size
        zbs = 0.01

        if not hasattr(self, "_coef_acceleration_data"):
            # Order the merger events by decreasing redshifts
            mevents = list(reversed(self.merger_events))
            chi_z = [self._chi_at_zidx(zidx, mevents)
                     for zidx in range(len(mevents))]
            zgrid = np.arange(0.0, self.zmax+zbs, step=zbs)
            chigrid = np.zeros(zgrid.shape)
            for (chi_, zbegin, zend) in chi_z:
                # NOTE: zbegin > zend
                mask = (zgrid <= zbegin) & (zgrid >= zend)
                chigrid[mask] += chi_
            self._coef_acceleration_data = (zgrid, chigrid)

        zgrid, chigrid = self._coef_acceleration_data
        # XXX: force a minimal value instead of zero or too small
        chi_min = 1 / (10 * self.cosmo.age0)

        try:
            zi = np.where(z < zgrid)[0][0]
            slope = (chigrid[zi]-chigrid[zi-1]) / (zgrid[zi]-zgrid[zi-1])
            chi = (z-zgrid[zi]) * slope + chigrid[zi]
        except IndexError:
            # Given z >= zgrid[-1]
            chi = 0.0

        if chi > chi_min:
            return chi
        else:
            return chi_min

    def _chi_at_zidx(self, zidx, mevents):
        """
        Calculate electron-acceleration coefficient at the specified
        merger event which is specified with a redshift index.

        Parameters
        ----------
        zidx : int
            Index of the redshift where to calculate the coefficient.
        mevents : list[dict]
            List of dictionaries that records all the merger events
            of the main cluster.
            NOTE:
            The merger events should be ordered by increasing time
            (or decreasing redshifts).

        Returns
        -------
        chi : float
            The calculated electron-acceleration coefficient.
            (unit: Gyr^-1)
        zbegin : float
            The redshift when the merger begins
        zend : float
            The redshift when the merger ends
            NOTE: zbegin > zend

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(40)
        """
        redshifts = np.array([ev["z"] for ev in mevents])
        zbegin = mevents[zidx]["z"]
        M_main = mevents[zidx]["M_main"]
        M_sub = mevents[zidx]["M_sub"]
        t_crossing = self._time_crossing(M_main, M_sub, zbegin)
        zend = self._z_end(zbegin, t_crossing)
        try:
            zend_idx = np.where(redshifts < zend)[0][0]
        except IndexError:
            # Specified redshift already the last/smallest one
            zend_idx = zidx + 1
        #
        coef = 2.23e-16 * self.eta_t / (self.radius/500)**3  # [s^-1]
        coef *= AUC.Gyr2s  # [Gyr^-1]
        chi = 0.0
        for ev in mevents[zidx:zend_idx]:
            M_main = ev["M_main"]
            M_sub = ev["M_sub"]
            z = ev["z"]
            R_vir = self._radius_virial(M_main, z)
            rs = self._radius_stripping(M_sub, M_main, z)
            kT = self.kT_mass(M_main)
            term1 = ((M_main+M_sub)/2e15 * (2.6e3/R_vir)) ** (3/2)
            term2 = (rs/500)**2 / np.sqrt(kT/7)
            if rs <= self.radius:
                term3 = 1.0
            else:
                term3 = (self.radius/rs) ** 2
            chi += coef * term1 * term2 * term3
        return (chi, zbegin, zend)

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
            term1 = (self.injection_index-2) * self.eta_e
            term2 = e_th / (self.pmin * self.mec * AC.c)  # [cm^-3]
            term3 = 1.0 / (self.cosmo.age0 * self.pmin)  # [Gyr^-1 mec^-1]
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
        z = self.cosmo.redshift(t)
        chi = self._coef_acceleration(z)  # [Gyr^-1]
        # NOTE: Cassano & Brunetti's formula misses a factor of 2.
        Dpp = chi * p**2 / 4  # [mec^2/Gyr]
        return Dpp

    def fp_advection(self, p, t):
        """
        Advection term/coefficient for the Fokker-Planck equation,
        which describes a systematic tendency for upward or downard
        drift of particles.

        This term is also called the "generalized cooling function" by
        Donnert & Brunetti (2014), which includes all relevant energy
        loss functions and the energy gain function due to turbulence.

        Returns
        -------
        Hp : float
            Advection coefficient
            Unit: [mec/Gyr]

        References
        ----------
        [1] Donnert & Brunetti 2014, MNRAS, 443, 3564
            http://adsabs.harvard.edu/abs/2014MNRAS.443.3564D
            Eq.(15)
        [2] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eqs.(30,36,38,39)
        """
        Hp = (abs(self._dpdt_ion(p, t)) +
              abs(self._dpdt_rad(p, t)) -
              (self.fp_diffusion(p, t) * 2 / p))
        return Hp

    def _dpdt_ion(self, p, t):
        """
        Energy loss through ionization and Coulomb collisions.

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(38)
        """
        z = self.cosmo.redshift(t)
        n_th = self._n_thermal(self.M0, z)
        coef = -3.3e-29 * AUC.Gyr2s / self.mec  # [mec/Gyr]
        dpdt = coef * n_th * (1 + np.log(p/n_th) / 75)
        return dpdt

    def _dpdt_rad(self, p, t):
        """
        Energy loss via synchrotron emission and IC scattering off the CMB.

        References
        ----------
        [1] Cassano & Brunetti 2005, MNRAS, 357, 1313
            http://adsabs.harvard.edu/abs/2005MNRAS.357.1313C
            Eq.(39)
        """
        z = self.cosmo.redshift(t)
        coef = -4.8e-4 * AUC.Gyr2s / self.mec  # [mec/Gyr]
        dpdt = (coef * (p*self.mec)**2 *
                ((self.magnetic_field/3.2)**2 + (1+z)**4))
        return dpdt

    @property
    def e_thermal(self):
        """
        Calculate the present-day thermal energy density of the ICM.

        Returns
        -------
        e_th : float
            Energy density of the ICM (unit: erg/cm^3)
        """
        mass = self.M0
        f_baryon = self.cosmo.Ob0 / self.cosmo.Om0
        kT = self.kT_mass(mass)  # [keV]
        N = mass * AUC.Msun2g * f_baryon / (AC.mu * AC.u)
        E_th = kT*AUC.keV2erg * N  # [erg]
        Rv = self._radius_virial(mass) * AUC.kpc2cm  # [cm]
        V = (4*np.pi / 3) * Rv**3  # [cm^3]
        e_th = E_th / V  # [erg/cm^3]
        return e_th

    def _n_thermal(self, mass, z=0.0):
        """
        Calculate the present-day number density of the ICM thermal plasma.

        Parameters
        ----------
        mass : float
            Mass (unit: Msun) of the cluster
        z : float
            Redshift

        Returns
        -------
        n_th : float
            Number density of the ICM (unit: cm^-3)
        """
        f_baryon = self.cosmo.Ob0 / self.cosmo.Om0
        N = mass * AUC.Msun2g * f_baryon / (AC.mu * AC.u)
        Rv = self._radius_virial(mass, z) * AUC.kpc2cm  # [cm]
        V = (4*np.pi / 3) * Rv**3  # [cm^3]
        n_th = N / V  # [cm^-3]
        return n_th
