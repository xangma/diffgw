"""Scalar 220 Kerr QNM reference from Cook--Zalutskiy (2014), Eqs. 23--43.

This intentionally uses mpmath and the n=0 Pincherle continued fraction with
a zero tail, rather than the production solver's inverted, asymptotic-tail
fraction. Angular matrix elements come from independent Wigner-d quadrature.
Only moderate prograde spins are in scope; depth and basis convergence must be
checked by callers.
"""

import mpmath as mp

from diffgw_reference.kerr_angular import spherical_harmonic


class ScalarQNM220:
    """Solve the fundamental positive-frequency spin -2, (ell,m)=(2,2) mode."""

    def __init__(self, *, ell_max=8, order=32, dps=35):
        if ell_max < 5 or order < 2 * ell_max + 2:
            raise ValueError("Angular basis or quadrature is too small")
        self.ell_max = ell_max
        self.order = order
        self.dps = dps
        with mp.workdps(dps):
            nodes, weights = mp.gauss_quadrature(order, "legendre")
            degrees = range(2, ell_max + 1)
            basis = [[spherical_harmonic(j, 2, mp.acos(x)) for j in degrees]
                     for x in nodes]
            size = len(degrees)
            self.x1 = mp.matrix(size)
            self.x2 = mp.matrix(size)
            for i in range(size):
                for j in range(size):
                    self.x1[i, j] = 2 * mp.pi * sum(
                        weights[k] * nodes[k] * mp.conj(basis[k][i]) * basis[k][j]
                        for k in range(order))
                    self.x2[i, j] = 2 * mp.pi * sum(
                        weights[k] * nodes[k] ** 2 * mp.conj(basis[k][i]) * basis[k][j]
                        for k in range(order))

    def angular_eigenvalue(self, c):
        """Return A(c), selected by proximity to the isolated A(0)=4 branch."""
        with mp.workdps(self.dps):
            if c == 0:
                return mp.mpc(4)
            size = self.ell_max - 1
            matrix = mp.matrix(size)
            for i in range(size):
                for j in range(size):
                    matrix[i, j] = ((i + 2) * (i + 3) - 2 if i == j else 0)
                    matrix[i, j] -= 4 * c * self.x1[i, j] + c * c * self.x2[i, j]
            return min(mp.eig(matrix, left=False, right=False), key=lambda z: abs(z - 4))

    def radial_residual(self, spin, omega, *, depth=400):
        """Evaluate Eq. 42 using Eq. 39 backward from the zero tail at depth."""
        if not 0 <= spin < 1 or depth < 20:
            raise ValueError("Require 0 <= spin < 1 and depth >= 20")
        with mp.workdps(self.dps):
            a, w = mp.mpf(spin), mp.mpc(omega)
            b = mp.sqrt(1 - a * a)
            sigma_plus = (2 * w * (1 + b) - 2 * a) / (2 * b)
            sigma_minus = (2 * w * (1 - b) - 2 * a) / (2 * b)
            xi, eta = 2 - 1j * sigma_plus, 2 + 1j * sigma_minus
            p = 1j * b * w
            alpha = 1 - 4j * w
            gamma, delta = -1 + 2 * eta, -1 + 2 * xi
            angular = self.angular_eigenvalue(a * w)
            sigma = (angular + a * a * w * w - 8 * w * w
                     + p * (2 * alpha + gamma - delta)
                     + (-1 - (gamma + delta) / 2)
                     * (-2 + (gamma + delta) / 2))
            d0 = delta
            d1 = 4 * p - 2 * alpha + gamma - delta - 2
            d2 = 2 * alpha - gamma + 2
            d3 = alpha * (4 * p - delta) - sigma
            d4 = alpha * (alpha - gamma + 1)

            ratio = mp.mpc(0)
            for n in range(depth - 1, -1, -1):
                k = n + 1
                ak = (k + 1) * (k + d0)
                bk = -2 * k * k + (d1 + 2) * k + d3
                gk = k * k + (d2 - 3) * k + d4 - d2 + 2
                ratio = -gk / (bk + ak * ratio)
            return d3 + d0 * ratio

    def root(self, spin, *, depth=400, seed=0.4-0.09j):
        """Find the 220 root near a continued fundamental-mode seed."""
        with mp.workdps(self.dps):
            solution = mp.findroot(
                lambda w: self.radial_residual(spin, w, depth=depth),
                (mp.mpc(seed), mp.mpc(seed) + mp.mpc('0.002', '-0.001')),
                tol=mp.mpf('1e-25'), maxsteps=35)
            return solution, self.angular_eigenvalue(mp.mpf(spin) * solution)
