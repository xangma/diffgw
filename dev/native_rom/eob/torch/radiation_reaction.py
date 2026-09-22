"""Historical 1.5PN prototype radiation reaction; development branch only."""

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
