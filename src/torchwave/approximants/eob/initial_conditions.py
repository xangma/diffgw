"""Effective-One-Body Initial Conditions for Quasicircular Inspirals.

Strict Clean-Room Implementation based on published peer-reviewed literature:
- Buonanno, Chen, Damour (2006), Phys. Rev. D 74, 104005 (arXiv:gr-qc/0508067)
- Pan et al. (2011), Phys. Rev. D 84, 124052 (arXiv:1106.1021)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

from typing import Tuple
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction


def find_quasicircular_initial_conditions(
    hamiltonian: EOBHamiltonian,
    radiation_reaction: EOBRadiationReaction,
    omega_start: torch.Tensor,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve for quasicircular initial state (r_0, pr_star_0, pphi_0) at target orbital frequency omega_start.

    Args:
        hamiltonian: Initialized EOBHamiltonian
        radiation_reaction: Initialized EOBRadiationReaction
        omega_start: Target starting dimensionless orbital frequency M * Omega_0

    Returns:
        (r_0, pr_star_0, pphi_0) tensors
    """
    # 1. Relativistic Schwarzschild initial guess: r ~ omega^(-2/3), pphi ~ sqrt(r / (1 - 3/r))
    omega_clamped = torch.clamp(omega_start, min=1e-5)
    r = (omega_clamped ** (-2.0 / 3.0)).clone()
    pphi = torch.sqrt(r / torch.clamp(1.0 - 3.0 / r, min=0.1)).clone()

    # 2. 2x2 Newton-Raphson iteration to solve simultaneously:
    # res_omega = (1 / H_real) * (dH_eff / dpphi) - omega_start = 0
    # res_circ  = dH_eff / dr = 0
    for _ in range(max_iter):
        r_var = r.detach().clone().requires_grad_(True)
        pphi_var = pphi.detach().clone().requires_grad_(True)
        pr_zero = torch.zeros_like(r)

        h_eff = hamiltonian.effective_hamiltonian(r_var, pr_zero, pphi_var)
        h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)

        grads = torch.autograd.grad(
            outputs=h_eff.sum(),
            inputs=[r_var, pphi_var],
            create_graph=True,
        )
        dh_dr = grads[0]
        dh_dpphi = grads[1]

        omega = dh_dpphi / h_real
        res_omega = omega - omega_start
        res_circ = dh_dr

        max_err = torch.max(torch.abs(res_omega) + torch.abs(res_circ)).item()
        if max_err < tol:
            break

        # Compute full 2x2 Jacobian
        j11, j12 = torch.autograd.grad(res_omega.sum(), [r_var, pphi_var], retain_graph=True)
        j21, j22 = torch.autograd.grad(res_circ.sum(), [r_var, pphi_var])

        det = j11 * j22 - j12 * j21
        delta_r = -(j22 * res_omega - j12 * res_circ) / (det + 1e-15)
        delta_pphi = -(-j21 * res_omega + j11 * res_circ) / (det + 1e-15)

        r = torch.clamp(r + delta_r.detach(), min=3.0)
        pphi = torch.clamp(pphi + delta_pphi.detach(), min=0.1)

    # 3. Post-adiabatic radiation-reaction radial momentum correction
    # Buonanno, Chen, Damour (2006) Phys. Rev. D 74, 104005 (arXiv:gr-qc/0508067) Eq. (4.7)
    # pr_*_0 = F_phi / (xi(r) * dp_phi/dr)
    _, f_phi = radiation_reaction.radiation_reaction_forces(omega_start, torch.zeros_like(r), pphi)
    _, _, xi = hamiltonian.metric_potentials(r)
    dp_phi_dr = 0.5 / torch.sqrt(torch.clamp(r, min=1.0))
    pr_star_corr = f_phi / (xi * dp_phi_dr + 1e-15)

    return r.detach(), pr_star_corr.detach(), pphi.detach()
