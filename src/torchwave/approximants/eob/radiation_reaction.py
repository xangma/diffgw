"""Effective-One-Body Dissipative Radiation Reaction Forces and Gravitational Flux.

The default is a low-order prototype. The explicit factorized class uses
equations reconstructed from the following primary publications:
- Buonanno & Damour (2000), Phys. Rev. D 62, 064015 (arXiv:gr-qc/0001013)
- Damour, Iyer, Nagar (2009), Phys. Rev. D 79, 064004 (arXiv:0811.2069)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
- Pompili et al. (2023), Phys. Rev. D 108, 124035 (arXiv:2303.18039)
"""

from typing import Tuple
import torch

from torchwave.constants import PI


class EOBRadiationReaction:
    """Prototype 1.5PN flux and normalized dissipative forces."""

    def __init__(self, nu: torch.Tensor, chi1: torch.Tensor, chi2: torch.Tensor):
        self.nu = nu
        self.chi1 = chi1
        self.chi2 = chi2

    def energy_flux(self, omega: torch.Tensor) -> torch.Tensor:
        """Compute the total gravitational wave energy flux dE/dt.

        Leading Newtonian quadrupole term:
            F_Newt = (32/5) * nu^2 * v^10
        augmented with relativistic post-Newtonian resummation factor hat{F}(v).
        """
        nu = self.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)

        # PN velocity parameter v_Omega = (M * Omega)^(1/3)
        v = torch.clamp(omega, min=1e-8) ** (1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v10 = (v2 * v2) * (v2 * v2) * v2

        # Resummed relativistic flux factor hat{F}(v)
        # Leading 1PN, 1.5PN tail terms
        f_1pn = -(1247.0 / 336.0) - (35.0 / 12.0) * nu
        f_15pn = 4.0 * PI

        flux_factor = 1.0 + f_1pn * v2 + f_15pn * v3
        flux_factor = torch.clamp(flux_factor, min=0.1)

        f_newtonian = (32.0 / 5.0) * (nu * nu) * v10
        total_flux = f_newtonian * flux_factor
        return total_flux

    def radiation_reaction_forces(
        self,
        omega: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
        *,
        r=None,
        hamiltonian=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the radiation-reaction dissipative force vector (F_r_star, F_phi).

        F_phi = - (1 / (nu * Omega)) * dE/dt
        F_r_* = (pr_* / p_phi) * F_phi
        """
        flux = self.energy_flux(omega)
        safe_omega = torch.clamp(torch.abs(omega), min=1e-8)

        # Azimuthal dissipative force (angular momentum loss)
        # p_phi is L/(mu M), whereas energy_flux is normalized by total mass.
        # Damour & Nagar 0704.3550v2 Eq. (12): normalized torque scales as nu.
        nu = self.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)
        f_phi = -(flux / (nu * safe_omega))

        # Radial dissipative force (small during quasicircular inspiral)
        sign = torch.where(pphi < 0, -torch.ones_like(pphi), torch.ones_like(pphi))
        safe_pphi = torch.where(torch.abs(pphi) > 1e-6, pphi, sign * 1e-6)
        f_r_star = (pr_star / safe_pphi) * f_phi

        return f_r_star, f_phi


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
