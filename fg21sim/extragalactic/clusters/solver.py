# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Solve the Fokker-Planck equation to derive the time evolution
of the electron spectrum (or number density distribution).

References
----------
.. [park1996]
   Park & Petrosian 1996, ApJS, 103, 255
   http://adsabs.harvard.edu/abs/1996ApJS..103..255P
.. [donnert2014]
   Donnert & Brunetti 2014, MNRAS, 443, 3564
   http://adsabs.harvard.edu/abs/2014MNRAS.443.3564D
"""

import logging

import numpy as np


logger = logging.getLogger(__name__)


def TDMAsolver(a, b, c, d):
    """
    Tri-diagonal matrix algorithm (a.k.a Thomas algorithm) solver,
    which is much faster than the generic Gaussian elimination algorithm.

    a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i],
    where: a[0] = c[N-1] = 0

    Example
    -------
    >>> A = np.array([[10,  2, 0, 0],
                      [ 3, 10, 4, 0],
                      [ 0,  1, 7, 5],
                      [ 0,  0, 3, 4]], dtype=float)
    >>> a = np.array([     3, 1, 3], dtype=float)
    >>> b = np.array([10, 10, 7, 4], dtype=float)
    >>> c = np.array([ 2,  4, 5   ], dtype=float)
    >>> d = np.array([ 3,  4, 5, 6], dtype=float)
    >>> print(TDMAsolver(a, b, c, d))
    [ 0.14877589  0.75612053 -1.00188324  2.25141243]
    # compare against numpy linear algebra library
    >>> print(np.linalg.solve(A, d))
    [ 0.14877589  0.75612053 -1.00188324  2.25141243]

    References
    ----------
    [1] http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    Credit
    ------
    [1] https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    """
    # Number of equations
    nf = len(d)
    # Copy the input arrays
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1] / bc[it-1]
        bc[it] -= mc*cc[it-1]
        dc[it] -= mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il]*xc[il+1]) / bc[il]

    return xc


class FokkerPlanckSolver:
    """
    Solve the Fokker-Planck equation:

    ∂u(x,t)   ∂  /                ∂u(x) \            u(x,t)
    ------- = -- | B(x)u(x) + C(x)----- | + Q(x,t) - ------
       ∂t     ∂x \                  ∂x  /            T(x,t)

    u(x,t) : distribution/spectrum w.r.t. x at different times
    B(x,t) : advection coefficient
    C(x,t) : diffusion coefficient (>0)
    Q(x,t) : injection coefficient (>=0)
    T(x,t) : escape coefficient

    NOTE
    ----
    The no-flux boundary condition is used, and optional boundary fix
    may be applied.

    Parameters
    ----------
    xmin, xmax : float
        The minimum and maximum bounds of the X (spatial/momentum) axis.
    x_np : int
        Number of (logarithmic grid) points/cells along the X axis
    tstep : float
        Specify to use the constant time step for solving the equation.
    f_advection : function
        Function f(x,t) to calculate the advection coefficient B(x,t)
    f_diffusion : function
        Function f(x,t) to calculate the diffusion coefficient C(x,t)
    f_injection : function
        Function f(x,t) to calculate the injection coefficient Q(x,t)
    f_escape : function, optional
        Function f(x,t) to calculate the escape coefficient T(x,t)
    buffer_np : int, optional
        Number of grid points taking as the buffer region near the lower
        boundary.  The densities within this buffer region will be replaced
        by extrapolating an power law to avoid unphysical accumulations.
        This fix is ignored if this parameter is not specified.
        (This parameter is suggested to be about 5%-10% of ``x_np``.)

    NOTE
    ----
    All above functions should accept two parameters: ``(x, t)``,
    where ``x`` is an 1D float `~numpy.ndarray` representing the adopted
    logarithmic grid points along the spatial/energy axis, ``t`` is the
    time of each solving step.

    NOTE
    ----
    The diffusion coefficients (i.e., calculated by ``f_diffusion()``)
    should be *positive* (i.e., C(x) > 0), otherwise unstable or wrong
    results may occur, due to the current numerical scheme/algorithm
    adopted.
    """

    def __init__(self, xmin, xmax, x_np, tstep,
                 f_advection, f_diffusion, f_injection,
                 f_escape=None, buffer_np=None):
        self.xmin = xmin
        self.xmax = xmax
        self.x_np = x_np
        self.tstep = tstep
        self.f_advection = f_advection
        self.f_diffusion = f_diffusion
        self.f_injection = f_injection
        self.f_escape = f_escape
        self.buffer_np = buffer_np

    @property
    def x(self):
        """
        X values of the adopted logarithmic grid.
        """
        grid = np.logspace(np.log10(self.xmin), np.log10(self.xmax),
                           num=self.x_np)
        return grid

    @property
    def dx(self):
        """
        Values of dx[i] on the grid.

        dx[i] = (x[i+1] - x[i-1]) / 2

        NOTE
        ----
        Extrapolate the X grid by 1 point beyond each side, therefore
        avoid NaN for the first and last element of dx[i].
        Otherwise, the subsequent calculation of tridiagonal coefficients
        may be invalid for the boundary elements.

        References: Ref.[park1996],Eq.(8)
        """
        x = self.x  # log scale
        # Extrapolate the x grid by 1 point beyond each side
        ratio = x[1] / x[0]
        x2 = np.concatenate([[x[0]/ratio], x, [x[-1]*ratio]])
        dx_ = (x2[2:] - x2[:-2]) / 2
        return dx_

    @property
    def dx_phalf(self):
        """
        Values of dx[i+1/2] on the grid.

        dx[i+1/2] = x[i+1] - x[i]
        Thus the last element is NaN.

        References: Ref.[park1996],Eq.(8)
        """
        x = self.x
        dx_ = x[1:] - x[:-1]
        grid = np.concatenate([dx_, [np.nan]])
        return grid

    @property
    def dx_mhalf(self):
        """
        Values of dx[i-1/2] on the grid.

        dx[i-1/2] = x[i] - x[i-1]
        Thus the first element is NaN.
        """
        x = self.x
        dx_ = x[1:] - x[:-1]
        grid = np.concatenate([[np.nan], dx_])
        return grid

    @staticmethod
    def X_phalf(X):
        """
        Calculate the values at midpoints (+1/2) for the given quantity.

        X[i+1/2] = (X[i] + X[i+1]) / 2
        Thus the last element is NaN.

        References: Ref.[park1996],Eq.(10)
        """
        Xmid = (X[1:] + X[:-1]) / 2
        return np.concatenate([Xmid, [np.nan]])

    @staticmethod
    def X_mhalf(X):
        """
        Calculate the values at midpoints (-1/2) for the given quantity.

        X[i-1/2] = (X[i-1] + X[i]) / 2
        Thus the first element is NaN.
        """
        Xmid = (X[1:] + X[:-1]) / 2
        return np.concatenate([[np.nan], Xmid])

    @staticmethod
    def W(w):
        # References: Ref.[park1996],Eqs.(27,35)
        w = np.asarray(w)
        with np.errstate(invalid="ignore"):
            # Ignore NaN's
            w = np.abs(w)
            mask = (w < 0.1)  # Comparison on NaN gives False, as expected
        W = np.zeros(w.shape) * np.nan
        W[mask] = 1.0 / (1 + w[mask]**2/24 + w[mask]**4/1920)
        W[~mask] = (w[~mask] * np.exp(-w[~mask]/2) /
                    (1 - np.exp(-w[~mask])))
        return W

    @staticmethod
    def bound_w(w, wmin=1e-8, wmax=1e3):
        """
        Bound the absolute values of ``w`` within [wmin, wmax], to avoid
        the underflow/overflow during later W/Wplus/Wminus calculations.
        """
        ww = np.array(w)
        with np.errstate(invalid="ignore"):
            # Ignore NaN's
            m1 = (np.abs(ww) < wmin)
            m2 = (np.abs(ww) > wmax)
        ww[m1] = wmin * np.sign(ww[m1])
        ww[m2] = wmax * np.sign(ww[m2])
        return ww

    def Wplus(self, w):
        # References: Ref.[park1996],Eq.(32)
        ww = self.bound_w(w)
        W = self.W(ww)
        Wplus = W * np.exp(ww/2)
        return Wplus

    def Wminus(self, w):
        # References: Ref.[park1996],Eq.(32)
        ww = self.bound_w(w)
        W = self.W(ww)
        Wminus = W * np.exp(-ww/2)
        return Wminus

    def tridiagonal_coefs(self, uc, tc, tstep):
        """
        Calculate the coefficients for the tridiagonal system of linear
        equations corresponding to the original Fokker-Planck equation.

        -a[i]*u[i-1] + b[i]*u[i] - c[i]*u[i+1] = r[i],
        where: a[0] = c[N-1] = 0

        NOTE
        ----
        When i=0 or i=N-1, b[i] is invalid due to X[-1/2] or X[N-1/2] are
        invalid. Therefore, b[0] and b[N-1] should be alternatively
        calculated with (e.g., no-flux) boundary condition considered.

        References: Ref.[park1996],Eqs.(16,18,34)
        """
        dt = tstep
        x = self.x
        dx = self.dx
        dx_phalf = self.dx_phalf
        dx_mhalf = self.dx_mhalf
        B = self.f_advection(x, tc)
        C = self.f_diffusion(x, tc)
        Q = self.f_injection(x, tc)
        #
        B_phalf = self.X_phalf(B)
        B_mhalf = self.X_mhalf(B)
        C_phalf = self.X_phalf(C)
        C_mhalf = self.X_mhalf(C)
        w_phalf = dx_phalf * B_phalf / C_phalf
        w_mhalf = dx_mhalf * B_mhalf / C_mhalf
        Wplus_phalf = self.Wplus(w_phalf)
        Wplus_mhalf = self.Wplus(w_mhalf)
        Wminus_phalf = self.Wminus(w_phalf)
        Wminus_mhalf = self.Wminus(w_mhalf)
        #
        a = (dt/dx) * (C_mhalf/dx_mhalf) * Wminus_mhalf
        a[0] = 0.0  # Fix a[0] which is NaN
        c = (dt/dx) * (C_phalf/dx_phalf) * Wplus_phalf
        c[-1] = 0.0  # Fix c[-1] which is NaN
        b = 1 + (dt/dx) * ((C_mhalf/dx_mhalf) * Wplus_mhalf +
                           (C_phalf/dx_phalf) * Wminus_phalf)
        # Calculate b[0] & b[-1], considering the no-flux boundary condition
        b[0] = 1 + (dt/dx[0]) * (C_phalf[0]/dx_phalf[0])*Wminus_phalf[0]
        b[-1] = 1 + (dt/dx[-1]) * (C_mhalf[-1]/dx_mhalf[-1])*Wplus_mhalf[-1]
        # Escape from the system
        if self.f_escape is not None:
            T = self.f_escape(x, tc)
            b += dt / T
        # Right-hand side
        r = dt * Q + uc
        return (a, b, c, r)

    def fix_boundary(self, uc):
        """
        Due to the no-flux boundary condition adopted, particles may
        unphysically pile up near the lower boundary.  Therefore, a
        buffer region spanning ``self.buffer_np`` cells is chosen, within
        which the densities are replaced by extrapolating from the upper
        density distribution as a power law, and the power-law index
        is determined by fitting to the data points of ``self.buffer_np``
        cells on the upper side of the buffer region.

        NOTE
        ----
        * Also fix the upper boundary in the same way.
        * Fix the boundaries only when the particles are piling up at the
          boundaries.

        References: Ref.[donnert2014],Sec.(3.3)
        """
        if self.buffer_np is None:
            return uc
        if (uc <= 0.0).sum() > 0:
            logger.warning("solved density has zero/negative values!")
            return uc

        x = self.x
        # Lower boundary
        ybuf = uc[:self.buffer_np]
        if ybuf[0] > ybuf[1]:
            # Particles are piling up at the lower boundary, to fix it...
            #
            # Power-law fit
            xp = x[self.buffer_np:(self.buffer_np*2)]
            yp = uc[self.buffer_np:(self.buffer_np*2)]
            pfit = np.polyfit(np.log(xp), np.log(yp), deg=1)
            xbuf = x[:self.buffer_np]
            ybuf = np.exp(np.polyval(pfit, np.log(xbuf)))
            uc[:self.buffer_np] = ybuf

        # Upper boundary
        ybuf = uc[(-self.buffer_np):]
        if ybuf[-1] > ybuf[-2]:
            # Particles are piling up at the upper boundary, to fix it...
            xp = x[(-self.buffer_np*2):(-self.buffer_np)]
            yp = uc[(-self.buffer_np*2):(-self.buffer_np)]
            pfit = np.polyfit(np.log(xp), np.log(yp), deg=1)
            xbuf = x[(-self.buffer_np):]
            ybuf = np.exp(np.polyval(pfit, np.log(xbuf)))
            uc[(-self.buffer_np):] = ybuf

        return uc

    def time_step(self):
        """
        Adaptively determine the time step for solving the equation.

        TODO/XXX
        """
        pass

    def solve_step(self, uc, tc, tstep=None):
        """
        Solve the Fokker-Planck equation by a single step.
        """
        if tstep is None:
            tstep = self.tstep
        a, b, c, r = self.tridiagonal_coefs(uc=uc, tc=tc, tstep=tstep)
        TDM_a = -a[1:]  # Also drop the first element
        TDM_b = b
        TDM_c = -c[:-1]  # Also drop the last element
        TDM_rhs = r
        t2 = tc + tstep
        u2 = TDMAsolver(TDM_a, TDM_b, TDM_c, TDM_rhs)
        u2 = self.fix_boundary(u2)
        return (u2, t2)

    def solve(self, u0, tstart, tstop):
        """
        Solve the Fokker-Planck equation from ``tstart`` to ``tstop``,
        with initial spectrum/distribution ``u0``.
        """
        uc = u0
        tc = tstart
        tstep = self.tstep
        logger.debug("Solving Fokker-Planck equation: " +
                     "time: %.3f - %.3f" % (tstart, tstop))
        nstep = int((tstop - tc) / tstep)
        logger.debug("Constant time step: %.3f (#%d steps)" % (tstep, nstep))
        i = 0
        while tc+tstep < tstop:
            i += 1
            logger.debug("[%d/%d] t=%.3f ..." % (i, nstep, tc))
            uc, tc = self.solve_step(uc=uc, tc=tc, tstep=tstep)
        # Last step
        tstep = tstop - tc
        logger.debug("Last step: t=%.3f (tstep=%.3f) ..." % (tc, tstep))
        uc, __ = self.solve_step(uc=uc, tc=tc, tstep=tstep)
        return uc


class FokkerPlanckTests:
    """
    Several Fokker-Planck equation test cases that have analytical solutions
    (hard-sphere approximation) to validate the above solver implementation.
    """
    xmin, xmax = 1e-4, 1e4
    x_np = 200
    x = np.logspace(np.log10(xmin), np.log10(xmax), x_np)
    tstep = 1e-3
    buffer_np = 20
    # Particle injection position/energy
    x_inj = 0.1

    def _f_injection(self, x, t):
        """
        Q(x,t) injection coefficient
        """
        idx = (self.x < self.x_inj).sum()
        dx = self.x[idx] - self.x[idx-1]
        sigma = dx / 2

        x = np.asarray(x)
        mu = (x - self.x_inj) / sigma
        coef = 1 / np.sqrt(2*np.pi * sigma**2)
        y = coef * np.exp(-0.5 * mu**2)
        return y

    def test1(self):
        """
        Fokker-Planck equation test case 1.

        WARNING
        -------
        The equations given by [park1996] and [donnert2014] both have a
        sign error about the advection term B(x).

        Usage
        -----
        >>> fpsolver = test1()
        >>> x = fpsolver.x
        >>> ts = [0, 0.2, 0.4, 0.7, 1.4, 2.7, 5.2, 10.0]
        >>> us = [None]*len(ts)
        >>> us[0] = np.zeros(x.shape)
        >>> for i, t in enumerate(ts[1:]):
        ...     tstart = ts[i]
        ...     tstop = ts[i+1]
        ...     print("* time: %.1f -> %.1f @ step: %.1e" %
        ...           (tstart, tstop, fpsolver.tstep))
        ...     us[i+1] = fpsolver.solve(u0=us[i], tstart=tstart, tstop=tstop)

        References
        ----------
        * [park1996], Eq.(22), Fig.(4)
        * [donnert2014], Eq.(34), Fig.(1:top-left)
        """
        def f_advection(x, t):
            # WARNING:
            # Both [park1996] and [donnert2014] got a "-1" for this term,
            # which should be "+1".
            return -x+1

        def f_diffusion(x, t):
            return x*x

        def f_injection(x, t):
            if t >= 0:
                return self._f_injection(x, t)
            else:
                return 0

        def f_escape(x, t):
            return 1

        fpsolver = FokkerPlanckSolver(xmin=self.xmin, xmax=self.xmax,
                                      x_np=self.x_np, tstep=self.tstep,
                                      f_advection=f_advection,
                                      f_diffusion=f_diffusion,
                                      f_injection=f_injection,
                                      f_escape=f_escape,
                                      buffer_np=self.buffer_np)
        return fpsolver

    def test2(self):
        """
        Fokker-Planck equation test case 2.

        References
        ----------
        * [park1996], Eq.(23), Fig.(2)
        * [donnert2014], Eq.(39), Fig.(1:bottom-left)
        """
        def f_advection(x, t):
            return -x

        def f_diffusion(x, t):
            return x*x

        def f_injection(x, t):
            if t >= 0:
                return self._f_injection(x, t)
            else:
                return 0

        def f_escape(x, t):
            return x

        fpsolver = FokkerPlanckSolver(xmin=self.xmin, xmax=self.xmax,
                                      x_np=self.x_np, tstep=self.tstep,
                                      f_advection=f_advection,
                                      f_diffusion=f_diffusion,
                                      f_injection=f_injection,
                                      f_escape=f_escape,
                                      buffer_np=self.buffer_np)
        return fpsolver

    def test3(self):
        """
        Fokker-Planck equation test case 3.

        References
        ----------
        * [park1996], Eq.(24), Fig.(3)
        * [donnert2014], Eq.(43), Fig.(1:bottom-right)
        """
        def f_advection(x, t):
            return -x**2

        def f_diffusion(x, t):
            return x**3

        def f_injection(x, t):
            if t == 0:
                return self._f_injection(x, 0) / self.tstep
            else:
                return 0

        def f_escape(x, t):
            return 1

        fpsolver = FokkerPlanckSolver(xmin=self.xmin, xmax=self.xmax,
                                      x_np=self.x_np, tstep=self.tstep,
                                      f_advection=f_advection,
                                      f_diffusion=f_diffusion,
                                      f_injection=f_injection,
                                      f_escape=f_escape,
                                      buffer_np=self.buffer_np)
        return fpsolver
