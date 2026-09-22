"""Equation-level EOB Hamiltonian components and shared derivative machinery.

Nonspinning3PNHamiltonian implements Damour--Nagar 0704.3550v2,
Eqs. (2)--(11), with no spin terms or waveform calibration. Its checks cover
Schwarzschild geodesics, the published PN expansion and the radial mass shell.
This is a conservative component, not a complete waveform approximant.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI
from jaxwave.conversions import as_array


class _HamiltonianBase(ABC):
    """Parameter bookkeeping and Hamilton equations; no model-specific physics."""

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

    @abstractmethod
    def metric_potentials(self, r):
        """Return A, B and the tortoise conversion xi for a supplied model."""
        raise NotImplementedError

    @abstractmethod
    def effective_hamiltonian(self, r, pr_star, pphi):
        """Return a model's reduced effective Hamiltonian."""
        raise NotImplementedError


class Nonspinning3PNHamiltonian(_HamiltonianBase):
    """Uncalibrated, strictly nonspinning 3PN conservative component."""

    def __init__(self, nu, delta=None):
        nu = as_array(nu, dtype=jnp.float64)
        zero = jnp.zeros_like(nu)
        super().__init__(nu, zero, zero, delta=delta)

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

    def effective_hamiltonian(self, r, pr_star, pphi):
        """DN Eq. (7), including the published quartic tortoise momentum."""
        nu = self.nu
        a, _, _ = self.metric_potentials(r)
        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        return jnp.sqrt(pr_star**2 + a * (1.0 + pphi**2 / r**2
                                             + z3 * pr_star**4 / r**2))


__all__ = ["Nonspinning3PNHamiltonian"]
