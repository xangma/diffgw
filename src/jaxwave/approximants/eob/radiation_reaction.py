"""Published factorized flux components and normalized dissipative forces.

These explicit selections are not a complete calibrated SEOBNRv4 flux.
"""

import jax.numpy as jnp

from jaxwave.conversions import as_array


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
    "FactorizedRadiationReaction",
]
