"""Clean TEOB state-to-mode adapter.

This module is the explicit seam between the reviewed nonspinning
``TEOBResumSHamiltonian`` and :class:`TEOBRadiationClean`.  It evaluates the
instantaneous quantities required by the radiation equations from a caller's
state ``(r, phi, p_r*, p_phi)`` and then calls the clean radiation component
directly.  The radiation component owns its TEOB tail factor, so this adapter
does not apply another carrier or tail correction.

Only the zero-spin, zero-tide Hamiltonian sector is supported.  The
Hamiltonian and radiation implementations remain separate: this file does
not select a trajectory, initial conditions, NQC, merger, or ringdown model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchwave.approximants.teobresums import TEOBResumSHamiltonian

from .teob_radiation_clean import TEOBRadiationClean


_UNSUPPORTED_INTRINSICS = (
    "chi1",
    "chi2",
    "lambda1",
    "lambda2",
    "kappa3_1",
    "kappa3_2",
)


def _require_supported_hamiltonian(hamiltonian: Any) -> TEOBResumSHamiltonian:
    """Validate the explicitly implemented Hamiltonian sector."""
    if not isinstance(hamiltonian, TEOBResumSHamiltonian):
        raise TypeError("hamiltonian must be a TEOBResumSHamiltonian")

    for name in _UNSUPPORTED_INTRINSICS:
        value = getattr(hamiltonian, name)
        if not isinstance(value, Tensor) or value.is_complex():
            raise TypeError(f"hamiltonian.{name} must be a real tensor")
        if not bool(torch.all(value == 0)):
            raise NotImplementedError(
                "teob_waveform_state_mode supports only zero spins and zero tides"
            )
    return hamiltonian


def _require_state(state: Any) -> Tensor:
    """Validate and return a real floating state with trailing size four."""
    if not isinstance(state, Tensor):
        raise TypeError("state must be a real float tensor with trailing axis (r,phi,p_r*,p_phi)")
    if state.ndim == 0 or state.shape[-1] != 4:
        raise ValueError("state must have trailing axis (r,phi,p_r*,p_phi)")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("state must use float32 or float64")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite")
    if bool((state[..., 0] <= 0).any()):
        raise ValueError("state radius must be positive")
    return state


def _require_mode(ell: Any, m: Any) -> tuple[int, int]:
    """Validate the signed spherical-harmonic mode requested by the caller."""
    if type(ell) is not int or type(m) is not int:
        raise TypeError("ell and m must be integers")
    if ell < 2 or ell > 8 or not (1 <= abs(m) <= ell):
        raise ValueError("requested mode must have 2<=ell<=8 and 1<=abs(m)<=ell")
    return ell, m


def _as_omega(value: Any, reference: Tensor) -> Tensor:
    """Convert a supplied orbital frequency without breaking tensor graphs."""
    if isinstance(value, Tensor) and value.is_complex():
        raise TypeError("Omega must be a real tensor or scalar")
    if isinstance(value, complex):
        raise TypeError("Omega must be a real tensor or scalar")
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.is_complex():
        raise TypeError("Omega must be a real tensor or scalar")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Omega must be finite")
    if bool((result <= 0).any()):
        raise ValueError("Omega must be positive")
    return result


def _radiation_inputs(
    state: Tensor,
    hamiltonian: TEOBResumSHamiltonian,
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
    """Build the current/circular energies, ``A'``, and mass fractions.

    ``Hcirc`` keeps the current angular momentum and replaces only
    ``p_r*`` by zero, as required by the clean non-Keplerian radius in
    ``TEOBRadiationClean``.  ``A'`` is the independent coordinate partial
    supplied by the Hamiltonian's clean derivative helper.
    """
    r, _phi, pr_star, pphi = state.unbind(-1)

    h_eff = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
    h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)

    h_eff_circ = hamiltonian.effective_hamiltonian(
        r, torch.zeros_like(pr_star), pphi,
    )
    h_circ = hamiltonian.real_hamiltonian_from_eff(h_eff_circ)
    a_prime = hamiltonian.metric_potential_derivative(r)

    # TEOBRadiationClean accepts normalized component masses.  Deriving these
    # from the Hamiltonian's mass fractions keeps the conversion explicit at
    # this adapter boundary and leaves any intrinsic batch axes intact.
    total_mass = hamiltonian.m1 + hamiltonian.m2
    m1 = hamiltonian.m1 / total_mass
    m2 = hamiltonian.m2 / total_mass
    return h_eff, h_real, h_circ, a_prime, (m1, m2)


def teob_waveform_state_mode(
    state: Tensor,
    hamiltonian: TEOBResumSHamiltonian,
    ell: int,
    m: int,
    *,
    Omega: Tensor | float | None = None,
    omega: Tensor | float | None = None,
    residual_phase: str = "zero-unpublished",
    nqc22: Tensor | float = 1.0,
) -> Tensor:
    """Return one clean TEOB mode from an explicit EOB state.

    Args:
        state: Tensor ending in ``(r, phi, p_r*, p_phi)``.  Leading axes are
            broadcast by the Hamiltonian and radiation stages.
        hamiltonian: A zero-spin, zero-tide ``TEOBResumSHamiltonian``.
        ell, m: Signed spherical-harmonic mode indices.
        Omega: Optional positive orbital frequency.  If omitted, the
            conservative Hamiltonian computes it from the state.  The
            lowercase ``omega`` spelling is accepted as an API compatibility
            alias; supplying both spellings is an error.
        residual_phase: Residual-phase policy passed to ``TEOBRadiationClean``.
        nqc22: Explicit multiplicative NQC factor for the (2,2) mode.

    ``TEOBRadiationClean.mode`` applies the TEOB tail gauge exactly once.  No
    additional tail, phase, or time shift is applied here.
    """
    state = _require_state(state)
    hamiltonian = _require_supported_hamiltonian(hamiltonian)
    ell, m = _require_mode(ell, m)

    if Omega is not None and omega is not None:
        raise TypeError("supply only one of Omega and omega")
    supplied_omega = Omega if Omega is not None else omega

    r, phi, pr_star, pphi = state.unbind(-1)
    if supplied_omega is None:
        _dr_dt, Omega_eval, _dpr_dt = hamiltonian.equations_of_motion_conservative(
            r, pr_star, pphi,
        )
    else:
        Omega_eval = _as_omega(supplied_omega, state)

    h_eff, h_real, h_circ, a_prime, masses = _radiation_inputs(state, hamiltonian)
    m1, m2 = masses
    radiation = TEOBRadiationClean(
        max(2, ell), residual_phase=residual_phase,
    )
    return radiation.mode(
        ell,
        m,
        r,
        Omega_eval,
        phi,
        pr_star,
        pphi,
        h_eff,
        h_real,
        h_circ,
        a_prime,
        m1,
        m2,
        nqc22=nqc22,
    )


# Short descriptive alias for callers following the existing ``teob22_mode``
# naming convention.
teob_state_mode = teob_waveform_state_mode


__all__ = ["teob_waveform_state_mode", "teob_state_mode"]
