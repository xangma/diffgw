"""Effective-One-Body Dissipative Radiation Reaction Forces and Gravitational Flux in JAX.

Clean-room implementations based on:
- Buonanno & Damour (2000), Phys. Rev. D 62, 064015 (arXiv:gr-qc/0001013)
- Damour, Iyer, Nagar (2009), Phys. Rev. D 79, 064004 (arXiv:0811.2069)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

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


class FactorizedRadiationReaction:
    """Circular factorized flux with normalized torque prescription."""

    def __init__(self, model: str):
        choices = {
            "din2009_nonspinning": (8, "none"),
            "pan2011_leading_spin": (4, "pan2011-leading-f"),
        }
        if model not in choices:
            raise ValueError("Unknown factorized radiation model")
        self.model = model
        self.lmax, self.spin_model = choices[model]
        try:
            from .factorized_modes import FactorizedModes
            self.modes = FactorizedModes(lmax=self.lmax, spin_model=self.spin_model)
        except ImportError:
            self.modes = None

    @staticmethod
    def _inputs(omega, pr_star, pphi, r, hamiltonian, phi=0.0):
        if r is None or hamiltonian is None:
            raise ValueError("Factorized radiation requires radius and Hamiltonian")
        r, pr_star, pphi = jnp.broadcast_arrays(
            as_array(r, dtype=jnp.float64),
            as_array(pr_star, dtype=jnp.float64),
            as_array(pphi, dtype=jnp.float64),
        )
        phi_arr = jnp.broadcast_to(as_array(phi, dtype=jnp.float64), r.shape)
        state = jnp.stack((r, phi_arr, pr_star, pphi), axis=-1)
        effective = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
        real = hamiltonian.real_hamiltonian_from_eff(effective)
        return (state, omega, effective, real, hamiltonian.m1, hamiltonian.m2)

    def energy_flux(self, omega, pr_star, pphi, *, r, hamiltonian):
        if self.modes is None:
            raise NotImplementedError("FactorizedModes not available in JAX")
        args = self._inputs(omega, pr_star, pphi, r, hamiltonian)
        return self.modes.flux(*args, chi1=hamiltonian.chi1, chi2=hamiltonian.chi2)

    def mode(self, ell, m, omega, pr_star, pphi, *, r, phi, hamiltonian):
        if self.modes is None:
            raise NotImplementedError("FactorizedModes not available in JAX")
        args = self._inputs(omega, pr_star, pphi, r, hamiltonian, phi)
        return self.modes.mode(ell, m, *args, chi1=hamiltonian.chi1, chi2=hamiltonian.chi2)

    def radiation_reaction_forces(self, omega, pr_star, pphi, *, r=None, hamiltonian=None):
        flux = self.energy_flux(omega, pr_star, pphi, r=r, hamiltonian=hamiltonian)
        nu = hamiltonian.nu
        if jnp.any(nu <= 0) or jnp.any(pphi == 0):
            raise ValueError("Dissipative factorized dynamics requires nu>0 and nonzero p_phi")
        torque = -flux / (nu * omega)
        return (pr_star / pphi) * torque, torque


__all__ = [
    "EOBRadiationReaction",
    "FactorizedRadiationReaction",
]
