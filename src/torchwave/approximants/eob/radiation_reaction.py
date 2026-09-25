"""Published factorized flux components and normalized dissipative forces.

These explicit selections are not a complete calibrated SEOBNRv4 flux.
"""

import torch



class FactorizedRadiationReaction:
    """Circular factorized flux with the DN normalized torque prescription.

    ``din2009_nonspinning`` sums all 35 modes through ell=8.
    ``pan2011_leading_spin`` retains only the published leading spin terms
    through ell=4. Neither selection is the calibrated SEOBNRv4 flux.
    Radius and the supplied Hamiltonian are required to evaluate the two
    rest-mass-inclusive energy source factors at each dynamical state.
    """

    def __init__(self, model: str):
        from .factorized_modes import FactorizedModes

        choices = {
            "din2009_nonspinning": (8, "none"),
            "pan2011_leading_spin": (4, "pan2011-leading-f"),
        }
        if model not in choices:
            raise ValueError("Unknown factorized radiation model")
        lmax, spin_model = choices[model]
        self.modes = FactorizedModes(lmax=lmax, spin_model=spin_model)

    @staticmethod
    def _inputs(omega, pr_star, pphi, r, hamiltonian, phi=0.):
        if r is None or hamiltonian is None:
            raise ValueError("Factorized radiation requires radius and Hamiltonian")
        r, pr_star, pphi, phi = torch.broadcast_tensors(
            r, pr_star, pphi, torch.as_tensor(phi, dtype=r.dtype, device=r.device)
        )
        state = torch.stack((r, phi, pr_star, pphi), dim=-1)
        effective = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
        real = hamiltonian.real_hamiltonian_from_eff(effective)
        return (state, omega, effective, real, hamiltonian.m1, hamiltonian.m2)

    def energy_flux(self, omega, pr_star, pphi, *, r, hamiltonian):
        args = self._inputs(omega, pr_star, pphi, r, hamiltonian)
        return self.modes.flux(*args, chi1=hamiltonian.chi1, chi2=hamiltonian.chi2)

    def mode(self, ell, m, omega, pr_star, pphi, *, r, phi, hamiltonian):
        args = self._inputs(omega, pr_star, pphi, r, hamiltonian, phi)
        return self.modes.mode(ell, m, *args, chi1=hamiltonian.chi1, chi2=hamiltonian.chi2)

    def radiation_reaction_forces(self, omega, pr_star, pphi, *, r=None, hamiltonian=None):
        flux = self.energy_flux(omega, pr_star, pphi, r=r, hamiltonian=hamiltonian)
        nu = hamiltonian.nu
        if torch.any(nu <= 0) or torch.any(pphi == 0):
            raise ValueError("Dissipative factorized dynamics requires nu>0 and nonzero p_phi")
        # DN 0704.3550 Eq. (12), with energy normalized by total mass and
        # momenta by reduced mass. No flux or frequency floor is applied.
        torque = -flux / (nu * omega)
        return (pr_star / pphi) * torque, torque
