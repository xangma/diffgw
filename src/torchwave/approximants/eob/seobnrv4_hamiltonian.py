"""Primary-equation reconstruction of the SEOBNRv4 Hamiltonian, for review.

This separate class reconstructs primary-paper equations. It is not selected
by the waveform generator and has not been certified against an implementation
oracle. See docs/reviews/2026-09-12-clean-room/pass-3-seobnrv4.md for sources
and the treatment of two inconsistent printed formulas.
"""

import math
from typing import NamedTuple

import torch

from torchwave.constants import EULER_GAMMA
from torchwave.approximants.eob.hamiltonian import _HamiltonianBase


class DeformedKerrPotentials(NamedTuple):
    delta_t: torch.Tensor
    delta_r: torch.Tensor
    delta_t_prime: torch.Tensor
    lambda_t: torch.Tensor
    lambda_t_prime: torch.Tensor
    xi: torch.Tensor


class SEOBNRv4Hamiltonian(_HamiltonianBase):
    """Aligned-spin Hamiltonian with the published v4 calibration coefficients.

    Units are M=1, momenta are divided by mu, and spins by M^2. The inherited
    constructor and conservative derivative wrapper provide only parameter
    bookkeeping and Hamilton's equations; all Hamiltonian terms are defined
    here. ``real_hamiltonian`` includes the rest mass, as the existing solver
    interface expects. ``reduced_binding_hamiltonian`` subtracts it.
    """

    def _parameter(self, value, r):
        while value.ndim < r.ndim:
            value = value.unsqueeze(-1)
        return value

    def spin_combinations(self, r):
        """Return S_Kerr/M^2 and sigma_*/M^2 without a mass-ratio division."""
        nu = self._parameter(self.nu, r)
        s1 = self._parameter(self.chi1 * self.m1.square(), r)
        s2 = self._parameter(self.chi2 * self.m2.square(), r)
        sigma_star = nu * self._parameter(self.chi1 + self.chi2, r)
        return s1 + s2, sigma_star, s1.square() + s2.square()

    def calibration_parameters(self, r):
        """Bohé et al. 1611.03703v3, Eqs. (2.8), (4.8), (4.12)-(4.14)."""
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        chi = a / (1.0 - 2.0 * nu)
        n2, n3 = nu.square(), nu.pow(3)
        c2, c3 = chi.square(), chi.pow(3)
        k_zero = 267.788247*n3 - 126.686734*n2 + 10.257281*nu + 1.733598
        k = (k_zero - 59.165806*c3*n3 - 0.426958*c3*nu + 1.436589*c3
             + 31.17459*c2*n3 + 6.164663*c2*n2 - 1.380863*c2
             - 27.520106*chi*n3 + 17.373601*chi*n2 + 2.268313*chi*nu
             - 1.62045*chi)
        d_so = (147.481449*c3*n2 - 568.651115*c3*nu + 66.198703*c3
                - 343.313058*c2*nu + 2495.293427*chi*n2 - 44.532373)
        d_ss = (528.511252*c3*n2 - 41.000256*c3*nu + 1161.780126*c2*n3
                - 326.324859*c2*n2 + 37.196389*chi*nu + 706.958312*n3
                - 36.027203*nu + 6.068071)
        return k, d_so, d_ss

    def radial_coefficients(self, r):
        """Steinhoff et al. 1608.01907v2, (A2), plus Bohé (2.3).

        The last two outputs are the constant and log(u) parts of Delta_5.
        Algebraically cancelling nu in (A2c) makes the test-mass limit regular.
        """
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        k, _, _ = self.calibration_parameters(r)
        m = k*nu - 1.0
        d0 = k*(k*nu - 2.0)
        d1 = -2.0*(d0 + k)*m
        d2 = 0.5*d1*(d1 - 4.0*m) - a.square()*d0*m.square()
        d3 = (-a.square()*d1*m.square() - d1.pow(3)/3.0 + d1.square()*m
              + d1*d2 - 2.0*m*(d2 - m))
        d4 = (8.0*(6.0*a.square()*(d1.square() - 2.0*d2)*m.square()
                    + 3.0*d1.pow(4) - 8.0*m*d1.pow(3) - 12.0*d1.square()*d2
                    + 12.0*d1*(2.0*d2*m + d3))
              + 48.0*d2.square() - 64.0*m*(3.0*d3 - 47.0*m)
              - 123.0*math.pi**2*m.square())/96.0
        d5 = m.square()*(
            -a.square()*(d1.pow(3) - 3.0*d1*d2 + 3.0*d3)/3.0
            + (d1.pow(4) - 4.0*d1.square()*d2 + 4.0*d1*d3
               + 2.0*d2.square() - 4.0*d4)/(2.0*m)
            - (d1.pow(5) - 5.0*d1.pow(3)*d2 + 5.0*d1.square()*d3
               + 5.0*d1*d2.square() - 5.0*d2*d3 - 5.0*d4*d1)/(5.0*m.square())
            + 2275.0*math.pi**2/512.0 + 128.0*EULER_GAMMA/5.0 - 4237.0/60.0
            + 256.0*math.log(2.0)/5.0 + nu*(41.0*math.pi**2/32.0 - 221.0/6.0)
        )
        return d0, d1, d2, d3, d4, d5, m.square()*64.0/5.0

    def deformed_kerr_potentials(self, r):
        """Log-resummed Delta_t, Delta_r and their analytic radial derivatives.

        The nu*log form is established in 1311.2544 Eq. (2) and 1608.01907
        Eq. (A1), and is required by the PN coefficients. Tortoise xi follows
        Pan et al. 0912.3466 Eq. (28): sqrt(Delta_t*Delta_r)/(r^2+a^2).
        """
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        k, _, _ = self.calibration_parameters(r)
        d0, d1, d2, d3, d4, d5_constant, d5_log = self.radial_coefficients(r)
        u = r.reciprocal()
        m = k*nu - 1.0
        d5 = d5_constant + d5_log*torch.log(u)
        polynomial_minus_one = u*(d1 + u*(d2 + u*(d3 + u*(d4 + u*d5))))
        polynomial_prime = (d1 + u*(2*d2 + u*(3*d3 + u*(4*d4
                            + u*(5*d5 + d5_log)))))
        factor = 1.0 + nu*d0 + nu*torch.log1p(polynomial_minus_one)
        # Expanded horizon product stays regular at a=0 and |a|=1.
        horizon_product = a.square()*u.square() + 2.0*u/m + m.reciprocal().square()
        delta_u = horizon_product*factor
        delta_u_prime = ((2.0*a.square()*u + 2.0/m)*factor
                        + horizon_product*nu*polynomial_prime/(1.0 + polynomial_minus_one))
        delta_t = r.square()*delta_u
        delta_t_prime = 2.0*r*delta_u - delta_u_prime
        d_inverse = 1.0 + torch.log1p(6.0*nu*u.square() + 2.0*(26.0-3.0*nu)*nu*u.pow(3))
        delta_r = delta_t*d_inverse
        varpi_squared = r.square() + a.square()
        lambda_t = varpi_squared.square() - a.square()*delta_t
        lambda_t_prime = 4.0*r*varpi_squared - a.square()*delta_t_prime
        xi = torch.sqrt(delta_t*delta_r)/varpi_squared
        return DeformedKerrPotentials(delta_t, delta_r, delta_t_prime,
                                     lambda_t, lambda_t_prime, xi)

    def metric_potentials(self, r):
        """Return -1/g^tt, g_rr and xi for the equatorial deformed metric."""
        metric = self.deformed_kerr_potentials(r)
        return (metric.delta_t*r.square()/metric.lambda_t,
                r.square()/metric.delta_r, metric.xi)

    def effective_spin(self, r, q_minus_one, radial_momentum_squared):
        """Zero-gauge spin map: 1107.2904 (51)-(52), 1311.2544 p. 2.

        radial_momentum_squared is g^rr*p_r^2 and q_minus_one is gamma^ij*p_i*p_j.
        The v2/v4 calibrated 4.5PN term multiplies S_Kerr, as explicitly stated
        in 1311.2544, rather than the earlier v1 sigma_* insertion.
        """
        nu = self._parameter(self.nu, r)
        a, sigma_star, _ = self.spin_combinations(r)
        _, d_so, _ = self.calibration_parameters(r)
        n2 = nu.square()
        q, p = q_minus_one, radial_momentum_squared
        star_1 = nu*(7.0/(6.0*r) + q/3.0 - 2.5*p)
        kerr_1 = nu*(-2.0/(3.0*r) + q/4.0 - 3.0*p)
        star_2 = ((353.0*nu - 27.0*n2)/(36.0*r.square()) + 5.0*n2*p.square()
                  - (23.0*nu + 3.0*n2)*q.square()/72.0
                  + (-103.0*nu + 60.0*n2)*q/(36.0*r)
                  + (16.0*nu - 21.0*n2)*p*q/12.0
                  + (47.0*nu - 54.0*n2)*p/(12.0*r))
        kerr_2 = (-(56.0*nu + 21.0*n2)/(9.0*r.square()) + 135.0*n2*p.square()/24.0
                  - 45.0*nu*q.square()/144.0
                  + (-109.0*nu + 51.0*n2)*q/(36.0*r)
                  + (6.0*nu - 39.0*n2)*p*q/24.0
                  - (16.0*nu + 147.0*n2)*p/(24.0*r))
        return sigma_star*(1.0 + star_1 + star_2) + a*(kerr_1 + kerr_2 + d_so*nu/r.pow(3))

    def effective_hamiltonian(self, r, pr_star, pphi):
        """Barausse–Buonanno 0912.3517, equatorial Eqs. (6.1)-(6.5)."""
        nu = self._parameter(self.nu, r)
        a, _, individual_spin_squares = self.spin_combinations(r)
        _, _, d_ss = self.calibration_parameters(r)
        metric = self.deformed_kerr_potentials(r)
        dt, dr, dt_prime, lam, lam_prime, xi = metric
        pr = pr_star/xi
        radial_squared = dr*pr.square()/r.square()
        q_minus_one = radial_squared + pphi.square()*r.square()/lam
        sqrt_q = torch.sqrt(1.0 + q_minus_one)
        b, j_radial = torch.sqrt(dt), torch.sqrt(dr)
        lapse = torch.sqrt(dt*r.square()/lam)
        nu_r = 0.5*(dt_prime/dt + 2.0/r - lam_prime/lam)
        b_r = dt_prime/(2.0*b) - b/j_radial
        omega = 2.0*a*r/lam
        omega_r = 2.0*a*(lam - r*lam_prime)/lam.square()
        s_star = self.effective_spin(r, q_minus_one, radial_squared)

        # Equation (6.3), with e^mu_tilde=r and M=1.
        g_so = lapse.square()/r*(
            -sqrt_q*j_radial*(b_r - 2.0*b*nu_r)
            + (r*lapse - b)*(sqrt_q + 1.0)
            + (b*nu_r - b_r)*j_radial
        )/(b.square()*(sqrt_q + 1.0)*sqrt_q)
        # Equation (6.4), divided by mu, with p_phi/mu=pphi.
        g_ss = (omega + b*omega_r*j_radial/(2.0*r*lapse)
                + (pphi.square() - b.square()*dr*pr.square()/(r.square()*lapse.square()))
                * lapse*omega_r*j_radial/(2.0*r*b*(sqrt_q + 1.0)*sqrt_q))
        # Strict 3PN truncation of the tortoise quartic term. See review note
        # concerning the inconsistent printed 1202.0790 Eq. (15).
        quartic = 2.0*nu*(4.0 - 3.0*nu)*pr_star.pow(4)/r.square()
        h_geodesic = omega*pphi + lapse*torch.sqrt(1.0 + q_minus_one + quartic)
        h_spin = (g_so*pphi*s_star + g_ss*s_star - s_star.square()/(2.0*r.pow(3))
                  + d_ss*nu*individual_spin_squares/r.pow(4))
        return h_geodesic + h_spin

    def real_hamiltonian_from_eff(self, h_eff):
        nu = self._parameter(self.nu, h_eff)
        return torch.sqrt(1.0 + 2.0*nu*(h_eff - 1.0))

    def reduced_binding_hamiltonian(self, r, pr_star, pphi):
        """(H_real-M)/mu, evaluated without cancellation or division by nu."""
        h_eff = self.effective_hamiltonian(r, pr_star, pphi)
        h_real = self.real_hamiltonian_from_eff(h_eff)
        return 2.0*(h_eff - 1.0)/(h_real + 1.0)
