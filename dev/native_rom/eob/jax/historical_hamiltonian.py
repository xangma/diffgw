"""Development-only SEOBNRv4 Hamiltonian convention from pinned LAL source."""

import jax.numpy as jnp

from jaxwave.approximants.eob.seobnrv4_hamiltonian import (
    DeformedKerrPotentials, SEOBNRv4Hamiltonian,
)


class HistoricalLogDerivative(SEOBNRv4Hamiltonian):
    """Reproduce LAL's internal Delta_t derivative without changing its value."""

    def deformed_kerr_potentials(self, r):
        metric = super().deformed_kerr_potentials(r)
        nu = self._parameter(self.nu, r)
        a, _, _ = self.spin_combinations(r)
        k, _, _ = self.calibration_parameters(r)
        _, d1, d2, d3, d4, d5_constant, d5_log = self.radial_coefficients(r)
        u = 1.0 / r
        m = k*nu - 1.0
        d5 = d5_constant + d5_log*jnp.log(u)
        polynomial_minus_one = u*(d1 + u*(d2 + u*(d3 + u*(d4 + u*d5))))
        horizon_product = a**2*u**2 + 2.0*u/m + 1.0/m**2
        missing = horizon_product*nu*d5_log*u**4/(1.0 + polynomial_minus_one)
        return DeformedKerrPotentials(
            metric.delta_t, metric.delta_r, metric.delta_t_prime + missing,
            metric.lambda_t, metric.lambda_t_prime - a**2*missing, metric.xi)
