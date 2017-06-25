# Copyright (c) 2017 Weitian LI <weitian@aaronly.me>
# MIT license

"""
Solve the Fokker-Planck equation to derive the time evolution
of the electron spectrum (or number density distribution).
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
    Solve the Fokker-Planck equation.

    ∂u(x,t)   ∂  /                ∂u(x) \            u(x,t)
    ------- = -- | B(x)u(x) + C(x)----- | + Q(x,t) - ------
       ∂t     ∂x \                  ∂x  /            T(x,t)

    u(x,t) : distribution/spectrum w.r.t. x at different times
    B(x,t) : advection coefficient
    C(x,t) : diffusion coefficient (>0)
    Q(x,t) : injection coefficient (>=0)
    T(x,t) : escape coefficient

    References
    ----------
    [1] Park & Petrosian 1996, ApJS, 103, 255
        http://adsabs.harvard.edu/abs/1996ApJS..103..255P
    [2] Donnert & Brunetti 2014, MNRAS, 443, 3564
        http://adsabs.harvard.edu/abs/2014MNRAS.443.3564D
    """

    def __init__(self, xmin, xmax, grid_num, buffer_np, tstep,
                 f_advection, f_diffusion, f_injection, f_escape=None):
        self.xmin = xmin
        self.xmax = xmax
        # Number of points on the logarithmic grid
        self.grid_num = grid_num
        # Number of grid points for the buffer region near the lower boundary
        self.buffer_np = buffer_np
        # Time step
        self.tstep = tstep
        # Function f(x,t) to calculate the advection coefficient B(x,t)
        self.f_advection = f_advection
        # Function f(x,t) to calculate the diffusion coefficient C(x,t)
        self.f_diffusion = f_diffusion
        # Function f(x,t) to calculate the injection coefficient Q(x,t)
        self.f_injection = f_injection
        # Function f(x,t) to calculate the escape coefficient T(x,t)
        self.f_escape = f_escape

    @property
    def x(self):
        """
        X values of the adopted logarithmic grid.
        """
        grid = np.logspace(np.log10(self.xmin), np.log10(self.xmax),
                           num=self.grid_num)
        return grid

    @property
    def dx(self):
        """
        Values of dx[i] on the grid.

        dx[i] = (x[i+1] - x[i-1]) / 2

        NOTE:
        Extrapolate the x grid by 1 point beyond each side, therefore
        avoid NaN for the first and last element of dx[i].
        Otherwise, the following calculation of tridiagonal coefficients
        may be invalid on the boundary elements.

        References: Ref.[1],Eq.(8)
        """
        x = self.x
        # Extrapolate the x grid by 1 point beyond each side
        x2 = np.concatenate([
            [x[0]**2/x[1]],
            x,
            [x[-1]**2/x[-2]],
        ])
        dx_ = (x2[2:] - x2[:-2]) / 2
        return dx_

    @property
    def dx_phalf(self):
        """
        Values of dx[i+1/2] on the grid.

        dx[i+1/2] = x[i+1] - x[i]
        Thus the last element is NaN.

        References: Ref.[1],Eq.(8)
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

        References: Ref.[1],Eq.(10)
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
        # References: Ref.[1],Eqs.(27,35)
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
        Bound the absolute values of w within [wmin, wmax].

        To avoid the underflow/overflow during later W/Wplus/Wminus
        calculations.
        """
        with np.errstate(invalid="ignore"):
            # Ignore NaN's
            m1 = (np.abs(w) < wmin)
            m2 = (np.abs(w) > wmax)
        ww = np.array(w)
        ww[m1] = wmin * np.sign(ww[m1])
        ww[m2] = wmax * np.sign(ww[m2])
        return ww

    def Wplus(self, w):
        # References: Ref.[1],Eq.(32)
        ww = self.bound_w(w)
        W = self.W(ww)
        Wplus = W * np.exp(ww/2)
        return Wplus

    def Wminus(self, w):
        # References: Ref.[1],Eq.(32)
        ww = self.bound_w(w)
        W = self.W(ww)
        Wminus = W * np.exp(-ww/2)
        return Wminus

    def tridiagonal_coefs(self, tc, uc):
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

        References: Ref.[1],Eqs.(16,18,34)
        """
        x = self.x
        dx = self.dx
        dx_phalf = self.dx_phalf
        dx_mhalf = self.dx_mhalf
        dt = self.tstep
        B = np.array([self.f_advection(x_, tc) for x_ in x])
        C = np.array([self.f_diffusion(x_, tc) for x_ in x])
        Q = np.array([self.f_injection(x_, tc) for x_ in x])
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
            T = np.array([self.f_escape(x_, tc) for x_ in x])
            b += dt / T
        # Right-hand side
        r = dt * Q + uc
        return (a, b, c, r)

    def fix_boundary(self, uc):
        """
        Truncate the lower end (i.e., near the lower boundary) of the
        distribution/spectrum and then extrapolate as a power law, in order
        to avoid the unphysical pile-up of electrons at the lower regime.

        References: Ref.[2],Sec.(3.3)
        """
        uc = np.asarray(uc)
        x = self.x
        # Calculate the power-law index
        xa = x[self.buffer_np]
        xb = x[self.buffer_np+1]
        ya = uc[self.buffer_np]
        yb = uc[self.buffer_np+1]
        if ya > 0 and yb > 0:
            # Truncate and extrapolate as a power law
            s = np.log(yb/ya) / np.log(xb/xa)
            uc[:self.buffer_np] = ya * (x[:self.buffer_np] / xa) ** s
        return uc

    def solve_step(self, tc, uc):
        """
        Solve the Fokker-Planck equation by a single step.
        """
        a, b, c, r = self.tridiagonal_coefs(tc=tc, uc=uc)
        TDM_a = -a[1:]  # Also drop the first element
        TDM_b = b
        TDM_c = -c[:-1]  # Also drop the last element
        TDM_rhs = r
        t2 = tc + self.tstep
        u2 = TDMAsolver(TDM_a, TDM_b, TDM_c, TDM_rhs)
        u2 = self.fix_boundary(u2)
        # Clear negative number densities
        # u2[u2 < 0] = 0
        return (t2, u2)

    def solve(self, u0, tstart, tstop):
        """
        Solve the Fokker-Planck equation from ``tstart`` to ``tstop``,
        with initial spectrum/distribution ``u0``.
        """
        uc = u0
        tc = tstart
        logger.info("Solving Fokker-Planck equation: " +
                    "time: %.3f - %.3f" % (tstart, tstop))
        nstep = (tstop - tc) / self.tstep
        i = 0
        while tc < tstop:
            i += 1
            logger.debug("[%d/%d] t=%.3f ..." % (i, nstep, tc))
            tc, uc = self.solve_step(tc, uc)
        return uc
