"""Historical 1.5PN prototype radiation reaction; development branch only."""

from typing import Tuple
import jax.numpy as jnp

from diffgw_core.constants import PI
from jaxwave.conversions import as_array


class EOBRadiationReaction:
    """Prototype 1.5PN flux and normalized dissipative forces in JAX."""

    def __init__(self, nu: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray):
        self.nu = as_array(nu, dtype=jnp.float64)
        self.chi1 = as_array(chi1, dtype=jnp.float64)
        self.chi2 = as_array(chi2, dtype=jnp.float64)

    def energy_flux(self, omega: jnp.ndarray) -> jnp.ndarray:
        """Compute the total gravitational wave energy flux dE/dt."""
        omega = as_array(omega, dtype=jnp.float64)
        v = jnp.maximum(omega, 1e-8) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v10 = (v2 * v2) * (v2 * v2) * v2

        f_1pn = -(1247.0 / 336.0) - (35.0 / 12.0) * self.nu
        f_15pn = 4.0 * PI

        flux_factor = 1.0 + f_1pn * v2 + f_15pn * v3
        flux_factor = jnp.maximum(flux_factor, 0.1)

        f_newtonian = (32.0 / 5.0) * (self.nu * self.nu) * v10
        total_flux = f_newtonian * flux_factor
        return total_flux

    def radiation_reaction_forces(
        self,
        omega: jnp.ndarray,
        pr_star: jnp.ndarray,
        pphi: jnp.ndarray,
        *,
        r=None,
        hamiltonian=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the radiation-reaction dissipative force vector (F_r_star, F_phi).

        F_phi = - (1 / (nu * Omega)) * dE/dt
        F_r_* = (pr_* / p_phi) * F_phi
        """
        omega = as_array(omega, dtype=jnp.float64)
        pr_star = as_array(pr_star, dtype=jnp.float64)
        pphi = as_array(pphi, dtype=jnp.float64)

        flux = self.energy_flux(omega)
        safe_omega = jnp.maximum(jnp.abs(omega), 1e-8)

        f_phi = -(flux / (self.nu * safe_omega))

        sign = jnp.where(pphi < 0, -jnp.ones_like(pphi), jnp.ones_like(pphi))
        safe_pphi = jnp.where(jnp.abs(pphi) > 1e-6, pphi, sign * 1e-6)
        f_r_star = (pr_star / safe_pphi) * f_phi

        return f_r_star, f_phi
