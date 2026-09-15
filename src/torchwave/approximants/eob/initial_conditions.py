"""Differentiable circular solve and leading adiabatic radial momentum.

Buonanno--Damour, arXiv:gr-qc/0001013v2, Eqs. (4.1), (4.13)-(4.16),
and Damour--Nagar, arXiv:0704.3550v2, Eqs. (6)-(11). This initializes the
supplied Hamiltonian/torque and is not a calibrated SEOBNRv4 initial-data fit.
"""

from typing import Tuple
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian, _ordinary_tensor_attributes
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction


@torch.inference_mode(False)
@torch.enable_grad()
def _solve_quasicircular_initial_conditions(
    hamiltonian: EOBHamiltonian,
    radiation_reaction: EOBRadiationReaction,
    omega_start: torch.Tensor,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve a stable circular orbit and its leading adiabatic inward momentum.

    Args:
        hamiltonian: Initialized EOBHamiltonian
        radiation_reaction: Initialized EOBRadiationReaction
        omega_start: Target starting dimensionless orbital frequency M * Omega_0

    Returns:
        (r_0, pr_star_0, pphi_0) tensors
    """
    hamiltonian = _ordinary_tensor_attributes(hamiltonian)
    radiation_reaction = _ordinary_tensor_attributes(radiation_reaction)
    omega_start = omega_start.clone()
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if torch.any(omega_start <= 0):
        raise ValueError("omega_start must be positive")
    omega_start = torch.broadcast_tensors(
        omega_start, hamiltonian.nu, hamiltonian.chi1, hamiltonian.chi2,
    )[0]

    # Schwarzschild circular orbit is the exact nu=0, zero-spin initial guess.
    r = omega_start ** (-2.0 / 3.0)
    pphi = torch.sqrt(r / torch.clamp(1.0 - 3.0 / r, min=0.1)).clone()

    # 2. 2x2 Newton-Raphson iteration to solve simultaneously:
    # res_omega = (1 / H_real) * (dH_eff / dpphi) - omega_start = 0
    # res_circ  = dH_eff / dr = 0
    for _ in range(max_iter):
        # Clones define independent partial-derivative coordinates without
        # severing the parameter/frequency graph carried by earlier iterates.
        r_var = r.clone().requires_grad_(True)
        pphi_var = pphi.clone().requires_grad_(True)
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

        # Compute full 2x2 Jacobian
        j11, j12 = torch.autograd.grad(
            res_omega.sum(), [r_var, pphi_var], create_graph=True,
        )
        j21, j22 = torch.autograd.grad(
            res_circ.sum(), [r_var, pphi_var], create_graph=True,
        )
        jacobian = torch.stack(
            (torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)),
            dim=-2,
        )
        residual = torch.stack((res_omega, res_circ), dim=-1)
        step = torch.linalg.solve(jacobian, -residual.unsqueeze(-1)).squeeze(-1)
        r = torch.clamp(r + step[..., 0], min=3.0)
        pphi = torch.clamp(pphi + step[..., 1], min=0.1)

        # A final Newton update is intentional even when the initial guess
        # already solves the equations: it supplies the implicit derivative
        # with respect to parameters absent from that initial guess.
        if torch.max(torch.abs(residual)).detach().item() < tol:
            break

    # Along H_eff,r(r, 0, j(r))=0, j'=-H_eff,rr/H_eff,rj.
    # Buonanno--Damour Eq. (4.13) gives rdot=F_phi/j'. Convert this
    # velocity to tortoise momentum with the linearized Hamilton equation,
    # rdot = [xi * H_eff,p*p* / H_real]_(p*=0) p* + O(p*^3).
    r_var = r.clone().requires_grad_(True)
    pphi_var = pphi.clone().requires_grad_(True)
    pr_zero = torch.zeros_like(r).requires_grad_(True)
    h_eff = hamiltonian.effective_hamiltonian(r_var, pr_zero, pphi_var)
    h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
    dh_dr, dh_dpr, dh_dj = torch.autograd.grad(
        h_eff.sum(), (r_var, pr_zero, pphi_var), create_graph=True,
    )
    final_residual = torch.stack((dh_dr, dh_dj / h_real - omega_start))
    effective_tol = max(tol, 32.0 * torch.finfo(r.dtype).eps)
    if not torch.all(torch.isfinite(final_residual)) or torch.any(final_residual.abs() > effective_tol):
        raise RuntimeError("Circular EOB initial conditions did not converge")
    d2h_dr2, d2h_drdj = torch.autograd.grad(
        dh_dr.sum(), (r_var, pphi_var), create_graph=True,
    )
    d2h_dpr2 = torch.autograd.grad(dh_dpr.sum(), pr_zero, create_graph=True)[0]
    dp_phi_dr = -d2h_dr2 / d2h_drdj
    if torch.any(dp_phi_dr <= 0):
        raise ValueError("Adiabatic EOB initial conditions require a stable circular orbit")
    _, f_phi = radiation_reaction.radiation_reaction_forces(
        omega_start, torch.zeros_like(r), pphi, r=r, hamiltonian=hamiltonian,
    )
    _, _, xi = hamiltonian.metric_potentials(r)
    velocity_per_momentum = xi * d2h_dpr2 / h_real
    pr_star_corr = f_phi / (velocity_per_momentum * dp_phi_dr)

    return r, pr_star_corr, pphi


def find_quasicircular_initial_conditions(
    hamiltonian: EOBHamiltonian,
    radiation_reaction: EOBRadiationReaction,
    omega_start: torch.Tensor,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (r, p_r*, p_phi) at a positive dimensionless orbital frequency.

    Solve the supplied Hamiltonian's stable circular branch, then convert its
    supplied normalized torque to leading adiabatic radial momentum. Gradients
    reach frequency and Hamiltonian parameters when requested. Failed solves
    and initial states on the unstable circular branch raise an exception.
    """
    # Hamiltonian extensions can add differentiable tidal/quadrupole inputs;
    # retain the solve graph for those, or for parameters of the supplied
    # torque, just as for the common mass/spin/frequency inputs.
    track_grad = torch.is_grad_enabled() and any(
        value.requires_grad for value in (
            omega_start, *vars(hamiltonian).values(), *vars(radiation_reaction).values(),
        ) if isinstance(value, torch.Tensor)
    )
    result = _solve_quasicircular_initial_conditions(
        hamiltonian, radiation_reaction, omega_start, max_iter, tol,
    )
    if track_grad:
        return result
    return tuple(value.detach() for value in result)
