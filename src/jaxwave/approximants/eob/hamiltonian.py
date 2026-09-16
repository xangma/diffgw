"""Prototype Effective-One-Body conservative Hamiltonian in JAX.

The nonspinning sector implements the uncalibrated 3PN model of Damour and
Nagar, PRD 76, 044003 (2007), arXiv:0704.3550v2, Eqs. (2)-(11).
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI
from jaxwave.conversions import as_array


class EOBHamiltonian:
    """Conservative Effective-One-Body Hamiltonian for aligned-spin binary black holes.

    Implements the energy mapping:
        H_real = M * sqrt(1 + 2 * nu * (H_eff / mu - 1))
    in geometric units where total mass M = 1 and reduced mass mu = nu.
    """

    def __init__(
        self,
        nu: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        delta: Optional[jnp.ndarray] = None,
    ):
        """Initialize EOB system parameters.

        Args:
            nu: Symmetric mass ratio eta = m1*m2/(m1+m2)^2
            chi1: Primary dimensionless aligned spin
            chi2: Secondary dimensionless aligned spin
            delta: Mass difference parameter (m1 - m2)/M = sqrt(1 - 4*nu)
        """
        self.nu = as_array(nu, dtype=jnp.float64)
        self.chi1 = as_array(chi1, dtype=jnp.float64)
        self.chi2 = as_array(chi2, dtype=jnp.float64)
        if delta is None:
            delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * self.nu, 0.0))
        self.delta = as_array(delta, dtype=jnp.float64)

        # Component mass fractions: m1/M = (1 + delta)/2, m2/M = (1 - delta)/2
        self.m1 = 0.5 * (1.0 + self.delta)
        self.m2 = 0.5 * (1.0 - self.delta)

        # Dimensionless spin variables: a1 = chi1 * m1, a2 = chi2 * m2
        self.a1 = self.chi1 * self.m1
        self.a2 = self.chi2 * self.m2

        self.a_plus = self.a1 + self.a2
        self.a_minus = self.a1 - self.a2

    def metric_potentials(self, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the 3PN Padé potentials and xi = dr/dr_* = sqrt(A/B).

        Damour--Nagar arXiv:0704.3550v2 Eqs. (2)-(3), (6) and Padé prescription.
        """
        u = 1.0 / r
        u2 = u * u
        u3 = u2 * u

        nu = self.nu
        a4 = (94.0 / 3.0) - (41.0 / 32.0) * (PI * PI)
        d1 = nu * (a4 + 4.0) / (8.0 - 2.0 * nu)
        a_potential = (1.0 + (d1 - 2.0) * u) / (
            1.0 + d1 * u + 2.0 * d1 * u2 + (4.0 * d1 - 2.0 * nu) * u3
        )
        d_potential = 1.0 / (
            1.0 + 6.0 * nu * u2 + 2.0 * nu * (26.0 - 3.0 * nu) * u3
        )
        b_potential = d_potential / a_potential
        xi = a_potential / jnp.sqrt(d_potential)

        return a_potential, b_potential, xi

    def gyro_factors(self, r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return prototype spin couplings."""
        u = 1.0 / r
        nu = self.nu
        g_plus = 2.0 - (9.0 / 8.0) * nu * u
        g_minus = (5.0 / 4.0) * nu * u
        return g_plus, g_minus

    def effective_hamiltonian(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the effective Hamiltonian H_eff / mu for equatorial aligned-spin orbits.

        Args:
            r: Dimensionless radial separation r / M
            pr_star: Conjugate momentum to tortoise coordinate r_* (p_{r_*} / mu)
            pphi: Orbital angular momentum (p_phi / (mu * M))

        Returns:
            Dimensionless effective Hamiltonian H_eff / mu
        """
        a_plus = self.a_plus
        a_minus = self.a_minus
        delta = self.delta
        nu = self.nu

        r2 = r * r
        r3 = r2 * r
        a_plus2 = a_plus * a_plus

        a_pot, _, _ = self.metric_potentials(r)
        g_plus, g_minus = self.gyro_factors(r)

        # 1. Odd-in-spin contribution (spin-orbit interaction)
        so_num = pphi * (g_plus * a_plus + g_minus * delta * a_minus)
        so_den = r3 + a_plus2 * (r + 2.0)
        h_odd = so_num / so_den

        # 2. Even-in-spin contribution
        spin_spin_term = (a_plus2 / r2) * (1.0 + 2.0 / r) * (pphi * pphi)
        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        angular_term = 1.0 + (pphi * pphi) / r2 + spin_spin_term
        h_even = jnp.sqrt(
            pr_star**2 + a_pot * (angular_term + z3 * (pr_star**4) / r2)
        )

        h_eff = h_odd + h_even
        return h_eff

    def real_hamiltonian_from_eff(self, h_eff: jnp.ndarray) -> jnp.ndarray:
        """Compute the real Hamiltonian H_real / M from dimensionless H_eff / mu."""
        radicand = 1.0 + 2.0 * self.nu * (h_eff - 1.0)
        return jnp.sqrt(jnp.maximum(radicand, 1e-6))

    def real_hamiltonian(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the real Hamiltonian H_real / M using the EOB energy map."""
        h_eff = self.effective_hamiltonian(r, pr_star, pphi)
        return self.real_hamiltonian_from_eff(h_eff)

    def equations_of_motion_conservative(
        self,
        r: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute conservative time derivatives (dr/dt, dphi/dt = Omega, dpr_star/dt).

        Evaluated via automatic differentiation:
            dr/dt = (xi(r) / H_real) * dH_eff / dpr_star
            dphi/dt = (1 / H_real) * dH_eff / dpphi = Omega
            (dpr_star/dt)_cons = -(xi(r) / H_real) * dH_eff / dr
        """
        r = as_array(r, dtype=jnp.float64)
        pr_star = as_array(pr_star, dtype=jnp.float64)
        pphi = as_array(pphi, dtype=jnp.float64)

        batch_shape = jnp.broadcast_shapes(
            r.shape, pr_star.shape, pphi.shape,
            self.nu.shape, self.chi1.shape, self.chi2.shape,
        )
        r_var = jnp.broadcast_to(r, batch_shape)
        pr_var = jnp.broadcast_to(pr_star, batch_shape)
        pphi_var = jnp.broadcast_to(pphi, batch_shape)

        def _sum_h_eff(r_in, pr_in, pphi_in):
            return jnp.sum(self.effective_hamiltonian(r_in, pr_in, pphi_in))

        dh_dr, dh_dpr, dh_dpphi = jax.grad(_sum_h_eff, argnums=(0, 1, 2))(r_var, pr_var, pphi_var)

        h_eff = self.effective_hamiltonian(r_var, pr_var, pphi_var)
        h_real = self.real_hamiltonian_from_eff(h_eff)
        _, _, xi = self.metric_potentials(r_var)

        dr_dt = (xi / h_real) * dh_dpr
        omega = (1.0 / h_real) * dh_dpphi
        dpr_star_dt = -(xi / h_real) * dh_dr

        return dr_dt, omega, dpr_star_dt


__all__ = [
    "EOBHamiltonian",
]
