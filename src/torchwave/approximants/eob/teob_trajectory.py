"""Standalone TEOBResumS trajectory quantities for NQC diagnostics.

The canonical state is ``(r, phi, p_r*, p_phi)``.  Damour--Nagar,
arXiv:0704.3550v2, Eq. (6), defines ``p_r*`` as the momentum conjugate to
the tortoise coordinate.  Their Hamilton equations give

    dr/dt = (xi/H_real) dH_eff/dp_r*
    dp_r*/dt = -(xi/H_real) dH_eff/dr.

This module applies the chain rule to the first equation along that
conservative flow to obtain ``d^2 r/dt^2``.  It deliberately does not select
an NQC extraction point or attach a waveform; those conventions are model
and paper dependent.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class TEOBResumSTrajectoryQuantities(NamedTuple):
    """Conservative quantities evaluated at one or more EOB states."""

    pr_star: torch.Tensor
    radial_velocity: torch.Tensor
    radial_acceleration: torch.Tensor
    pr_star_rate: torch.Tensor


def conservative_trajectory_quantities(hamiltonian, state: torch.Tensor):
    """Return canonical ``p_r*`` and conservative ``d²r/dt²``.

    ``state`` has final axis ``(r, phi, p_r*, p_phi)`` and may have arbitrary
    leading batch/sample axes.  Hamiltonian parameters broadcast against the
    state.  The returned acceleration excludes radiation reaction: with
    conservative ``dp_phi/dt = 0`` and no explicit ``phi`` dependence,

    ``ddot(r) = (partial_r rdot) rdot +
                (partial_p*r rdot) dot(p_r*)``.

    The local coordinate nodes are materialized before differentiating so a
    scalar state shared by several parameter rows still receives one
    acceleration per row.  This is an additive diagnostic API; it does not
    infer a merger time or NQC extraction convention.
    """
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch tensor")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("state must use float32 or float64")
    if state.ndim < 1 or state.shape[-1] != 4:
        raise ValueError("state must have final axis (r, phi, p_r*, p_phi)")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite")

    track_grad = torch.is_grad_enabled()
    with torch.inference_mode(False):
        result = _conservative_trajectory_quantities(hamiltonian, state)
    if track_grad:
        return result
    return result._replace(
        pr_star=result.pr_star.detach(),
        radial_velocity=result.radial_velocity.detach(),
        radial_acceleration=result.radial_acceleration.detach(),
        pr_star_rate=result.pr_star_rate.detach(),
    )


@torch.inference_mode(False)
def _conservative_trajectory_quantities(hamiltonian, state: torch.Tensor):
    with torch.enable_grad():
        r, pr_star, pphi = state[..., 0], state[..., 2], state[..., 3]
        r, pr_star, pphi = torch.broadcast_tensors(r, pr_star, pphi)
        # The Hamiltonian can add parameter batch axes to a scalar/shared state.
        with torch.no_grad():
            batch_shape = hamiltonian.effective_hamiltonian(r, pr_star, pphi).shape

        r_var = r.expand(batch_shape).clone().requires_grad_(True)
        pr_var = pr_star.expand(batch_shape).clone().requires_grad_(True)
        pphi_var = pphi.expand(batch_shape).clone().requires_grad_(True)
        radial_velocity, _, pr_star_rate = hamiltonian.equations_of_motion_conservative(
            r_var, pr_var, pphi_var)
        partials = torch.autograd.grad(
            radial_velocity.sum(),
            (r_var, pr_var, pphi_var),
            create_graph=True,
        )
        # The conservative flow has d p_phi/dt = 0.  The Hamiltonian has no phi
        # dependence, so the omitted phi term vanishes identically as well.
        radial_acceleration = partials[0] * radial_velocity + partials[1] * pr_star_rate
        result = TEOBResumSTrajectoryQuantities(
            pr_star=pr_var,
            radial_velocity=radial_velocity,
            radial_acceleration=radial_acceleration,
            pr_star_rate=pr_star_rate,
        )
    return result


__all__ = ["TEOBResumSTrajectoryQuantities", "conservative_trajectory_quantities"]
