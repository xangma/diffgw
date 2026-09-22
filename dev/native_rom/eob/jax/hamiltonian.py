"""Unvalidated historical spin Hamiltonian; development branch only."""

import jax.numpy as jnp
from typing import Tuple
from jaxwave.approximants.eob.hamiltonian import _HamiltonianBase, Nonspinning3PNHamiltonian


class EOBHamiltonian(_HamiltonianBase):
    """Prototype spin extensions retained for development comparisons."""

    metric_potentials = Nonspinning3PNHamiltonian.metric_potentials

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
