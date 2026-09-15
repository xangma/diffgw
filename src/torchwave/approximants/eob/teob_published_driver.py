"""Clean-room, nonspinning TEOB factorized (2,2) driver.

This is the bounded waveform stage described by Nagar et al.,
arXiv:1806.01772, Sec. II, with the factorized multipolar equations of
Damour, Iyer & Nagar, arXiv:0811.2069.  It consumes an EOB Hamiltonian and
trajectory field supplied by the caller; it does not choose an EOB potential
or initial conditions.

Included: the published nonspinning factorized 22 Newtonian prefactor,
source, residual amplitude, tail amplitude/phase, and residual phase.  The
same mode is used to compute the positive emitted flux.  Deliberately absent:
NQC corrections, calibrated TEOB potentials, tidal/self-spin terms, merger
attachment, and ringdown.  Consequently this module is an inspiral/plunge
stage, not a complete TEOBResumS model.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .factorized_modes import FactorizedModes
from .integrator import EOBTrajectoryIntegrator, FrequencyPeakTrajectory
from .hamiltonian import _ordinary_tensor_attributes


class TEOB22Result(NamedTuple):
    """Sampled factorized 22 mode and its continuous phase frequency."""

    mode: torch.Tensor
    frequency: torch.Tensor
    trajectory: FrequencyPeakTrajectory | None = None


def _inputs(state, hamiltonian, omega=None):
    if not isinstance(state, torch.Tensor) or state.ndim < 1 or state.shape[-1] != 4:
        raise TypeError("state must be a float tensor with trailing axis (r,phi,p_r*,p_phi)")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("state must use float32 or float64")
    r, phi, pr, pphi = state.unbind(-1)
    effective = hamiltonian.effective_hamiltonian(r, pr, pphi)
    real = hamiltonian.real_hamiltonian_from_eff(effective)
    if omega is None:
        _, omega, _ = hamiltonian.equations_of_motion_conservative(r, pr, pphi)
    return state, omega, effective, real, hamiltonian.m1, hamiltonian.m2


def teob22_mode(state, hamiltonian, *, omega=None,
                residual_phase: str = "published") -> torch.Tensor:
    """Return ``R h_22/M`` from one or more EOB states.

    The Hamiltonian must expose ``effective_hamiltonian``,
    ``real_hamiltonian_from_eff``, ``equations_of_motion_conservative``,
    ``m1`` and ``m2``.  Only zero aligned spins are accepted because the
    published stage intentionally has no spin or tidal extension.
    """
    if bool(torch.any(getattr(hamiltonian, "chi1", torch.tensor(0., device=state.device)) != 0)) or bool(
            torch.any(getattr(hamiltonian, "chi2", torch.tensor(0., device=state.device)) != 0)):
        raise NotImplementedError("teob22_mode currently implements the nonspinning stage only")
    args = _inputs(state, hamiltonian, omega)
    return FactorizedModes(2, residual_phase=residual_phase).mode(2, 2, *args)


def teob22_flux(state, hamiltonian, *, omega=None) -> torch.Tensor:
    """Return the positive emitted flux from the same factorized 22 mode.

    Higher multipoles are intentionally excluded, so this is a 22-only flux
    and must not be mistaken for the full TEOB radiation reaction flux.
    """
    if bool(torch.any(getattr(hamiltonian, "chi1", torch.tensor(0., device=state.device)) != 0)) or bool(
            torch.any(getattr(hamiltonian, "chi2", torch.tensor(0., device=state.device)) != 0)):
        raise NotImplementedError("teob22_flux currently implements the nonspinning stage only")
    args = _inputs(state, hamiltonian, omega)
    # The public flux sum includes every requested positive-m mode.  Select
    # (2,2) explicitly here; FactorizedModes(2).flux also includes (2,1).
    mode = FactorizedModes(2, residual_phase="published").mode(2, 2, *args)
    return (2 * args[1]).square() * mode.abs().square() / (8 * torch.pi)


def teob22_mode_and_frequency(state, hamiltonian, radiation_reaction=None):
    """Return ``(h22, -Im(dot(h22)/h22))`` along the supplied dynamics.

    With ``radiation_reaction=None`` the conservative Hamiltonian field is
    used.  Supplying a radiation-reaction object includes its dissipative
    terms, making the returned frequency a directional derivative of the
    complete state evolution rather than simply ``2*Omega``.
    """
    track_grad = torch.is_grad_enabled() and (state.requires_grad or any(
        isinstance(v, torch.Tensor) and v.requires_grad for v in vars(hamiltonian).values()))
    with torch.inference_mode(False), torch.enable_grad():
        local_ham = _ordinary_tensor_attributes(hamiltonian)
        local = state.clone()
        if radiation_reaction is None:
            hamiltonian_for_eval = local_ham
            dr, om, dpr = local_ham.equations_of_motion_conservative(
                local[..., 0], local[..., 2], local[..., 3])
            direction = torch.stack((dr, om, dpr, torch.zeros_like(dr)), dim=-1)
        else:
            hamiltonian_for_eval = local_ham
            direction = EOBTrajectoryIntegrator(local_ham, radiation_reaction).vector_field(local)
        shift = torch.zeros_like(local[..., 0], requires_grad=True)
        shifted = local + shift.unsqueeze(-1) * direction
        mode = teob22_mode(shifted, hamiltonian_for_eval)
        phase_derivative = torch.autograd.grad(
            torch.angle(mode).sum(), shift, create_graph=track_grad)[0]
        phase_derivative = -phase_derivative
    return (mode, phase_derivative) if track_grad else (mode.detach(), phase_derivative.detach())


def integrate_teob22(initial_state, hamiltonian, radiation_reaction, dt, num_steps,
                     *, post_peak_duration=0., richardson_extrapolation=False) -> TEOB22Result:
    """Integrate a caller-owned EOB trajectory and sample its factorized 22 mode."""
    trajectory = EOBTrajectoryIntegrator(hamiltonian, radiation_reaction).integrate_to_frequency_peak(
        initial_state, dt, num_steps, post_peak_duration=post_peak_duration,
        richardson_extrapolation=richardson_extrapolation)
    state = trajectory.state.movedim(0, -2)
    mode, frequency = teob22_mode_and_frequency(state, hamiltonian, radiation_reaction)
    return TEOB22Result(mode, frequency, trajectory)


__all__ = ["TEOB22Result", "teob22_mode", "teob22_flux",
           "teob22_mode_and_frequency", "integrate_teob22"]
