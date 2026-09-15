"""Explicit zero spin TEOB radial acceleration approximations.

This module records three different uses of the conservative radial
Hamilton equations.  It is intentionally an opt in diagnostic seam; it does
not replace :func:`teob_trajectory.conservative_trajectory_quantities` or
select an NQC convention at runtime.

For zero aligned spin the Hamiltonian used by the clean TEOB dependency is

``H_eff = sqrt(p**2 + A*(1 + j**2/r**2 + z3*p**4/r**2))``

and ``H = sqrt(1 + 2*nu*(H_eff - 1))``.  Writing the published radial
Hamilton equation as ``rdot = p*C`` gives

``C = xi*(1 + 2*z3*A*p**2/r**2)/(H*H_eff)``
``pdot = -xi/H * partial_r(H_eff)``.

The exact conservative helper differentiates ``rdot`` by the full chain
rule.  The named approximations here are:

``strict_p0``
    The leading radial momentum term.  ``C`` and ``pdot`` are both
    evaluated at ``p=0`` while the current ``r`` and ``j`` are retained.
    Thus ``a = -xi**2*(partial_r H_eff)|p=0 /
    (H|p=0)**2/(H_eff|p=0)``.

``mixed_current_p0`` and ``mixed_p0_current``
    Compatibility diagnostics that keep one factor at the current state and
    the other at ``p=0``.  They are exposed so a comparison cannot silently
    conflate these choices.

``frozen_mobility``
    A separate compatibility hypothesis: evaluate ``C`` and ``pdot`` at the
    current full state, then hold ``C`` fixed in the time derivative, giving
    ``a=C_current*pdot_current``.  At ``p=0`` its mobility is the analytic
    limit ``xi/(H0*H_eff0)``; it is never formed by dividing ``rdot`` by
    ``p``.  The cited equations motivate the factorization, but do not by
    themselves select this frozen-mobility approximation.

The displayed Hamiltonian and radial equation follow Nagar et al.,
arXiv:1806.01772v2, Sec. II (Eqs. (4)--(11)); the conservative acceleration
used for NQC is ``(ddot r)^(0)`` in Damour and Nagar, arXiv:1406.6913v1,
Eqs. (96)--(97), with the chain-rule construction made explicit in
Damour--Nagar--Bernuzzi, arXiv:1212.4357v2, Appendix A.  The API accepts a
scalar-intrinsic Hamiltonian and vectorizes only the supplied radial sample
axes, rejecting intrinsic parameter batches instead of broadcasting them
silently.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch import Tensor


class TEOBRadialAccelerationError(ValueError):
    """Raised when the requested zero-spin approximation is out of domain."""


@dataclass(frozen=True)
class TEOBRadialAccelerationTerms:
    """All explicitly named radial-acceleration branches at one profile.

    Every tensor has the broadcast shape of ``radius``, ``pr_star`` and
    ``p_phi``.  ``p0_*`` quantities use the analytic ``p=0`` branch, while
    ``current_*`` quantities use the supplied full radial momentum.
    """

    strict_p0: Tensor
    current_mobility_p0_force: Tensor
    p0_mobility_current_force: Tensor
    frozen_mobility: Tensor
    p0_radial_velocity: Tensor
    current_radial_velocity: Tensor
    p0_mobility: Tensor
    current_mobility: Tensor
    p0_pr_star_rate: Tensor
    current_pr_star_rate: Tensor
    p0_effective_energy: Tensor
    current_effective_energy: Tensor
    p0_real_energy: Tensor
    current_real_energy: Tensor
    p0_radial_velocity_radius_partial: Tensor
    p0_radial_velocity_pr_star_partial: Tensor

    @property
    def strict(self) -> Tensor:
        """Alias for the strict leading-``p`` branch."""

        return self.strict_p0

    @property
    def strict_leading_p(self) -> Tensor:
        """Descriptive alias for :attr:`strict_p0`."""

        return self.strict_p0

    @property
    def mixed_current_p0(self) -> Tensor:
        """Alias for current mobility with the strict ``p=0`` force."""

        return self.current_mobility_p0_force

    @property
    def mixed_p0_current(self) -> Tensor:
        """Alias for ``p=0`` mobility with the current force."""

        return self.p0_mobility_current_force

    @property
    def frozen(self) -> Tensor:
        """Alias for the frozen-mobility compatibility branch."""

        return self.frozen_mobility

    @property
    def current_force(self) -> Tensor:
        """The current conservative radial momentum rate."""

        return self.current_pr_star_rate

    @property
    def p0_force(self) -> Tensor:
        """The conservative radial momentum rate at ``p=0``."""

        return self.p0_pr_star_rate


Variant = Literal[
    "strict_p0",
    "mixed_current_p0",
    "mixed_p0_current",
    "frozen_mobility",
]


def _reference_tensor(*values: object) -> Tensor:
    for value in values:
        if isinstance(value, Tensor):
            if value.is_complex():
                raise TypeError("radial acceleration inputs must be real tensors")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("radial acceleration inputs must use float32 or float64")
            return value
    return torch.empty((), dtype=torch.get_default_dtype())


def _as_real(value: object, reference: Tensor, name: str) -> Tensor:
    if isinstance(value, complex) or (isinstance(value, Tensor) and value.is_complex()):
        raise TypeError(f"{name} must be a real tensor or scalar")
    if isinstance(value, Tensor):
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use float32 or float64")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise TypeError(f"{name} must share dtype and device with the reference")
        return value
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise TEOBRadialAccelerationError(f"{name} must be finite")


def _positive(name: str, value: Tensor) -> None:
    if bool((value <= 0).any()):
        raise TEOBRadialAccelerationError(f"{name} must be positive")


def _zero_if_none(value: Tensor | None, reference: Tensor) -> Tensor:
    return torch.zeros_like(reference) if value is None else value


def _check_hamiltonian(hamiltonian: object) -> None:
    required = (
        "nu",
        "chi1",
        "chi2",
        "metric_potentials",
        "effective_hamiltonian",
        "real_hamiltonian_from_eff",
        "equations_of_motion_conservative",
    )
    missing = [name for name in required if not hasattr(hamiltonian, name)]
    if missing:
        raise TypeError("hamiltonian is missing required TEOB methods/attributes: "
                        + ", ".join(missing))

    for name in ("nu", "chi1", "chi2"):
        value = getattr(hamiltonian, name)
        if not isinstance(value, Tensor):
            raise TypeError(f"hamiltonian.{name} must be a tensor")
        if value.is_complex() or value.dtype not in (torch.float32, torch.float64):
            raise TypeError("Hamiltonian intrinsic parameters must be real float32/float64 tensors")
        if value.ndim != 0:
            raise ValueError(
                "radial acceleration accepts scalar intrinsic parameters; "
                f"hamiltonian.{name} has shape {tuple(value.shape)}"
            )
        _finite(f"hamiltonian.{name}", value)

    if bool(getattr(hamiltonian, "chi1") != 0) or bool(getattr(hamiltonian, "chi2") != 0):
        raise NotImplementedError("radial acceleration approximations are zero-spin only")
    nu = getattr(hamiltonian, "nu")
    if bool((nu <= 0) | (nu > 0.25)):
        raise TEOBRadialAccelerationError("hamiltonian.nu must lie in (0, 0.25]")


def _prepare_inputs(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    _check_hamiltonian(hamiltonian)
    reference = _reference_tensor(radius, pr_star, p_phi, hamiltonian.nu)
    r = _as_real(radius, reference, "radius")
    p = _as_real(pr_star, reference, "pr_star")
    j = _as_real(p_phi, reference, "p_phi")
    r, p, j = torch.broadcast_tensors(r, p, j)
    _finite("radius", r)
    _finite("pr_star", p)
    _finite("p_phi", j)
    _positive("radius", r)

    # The state axes may be profiles, but intrinsic Hamiltonian parameters are
    # deliberately scalar.  Check the Hamiltonian's output shape before
    # cloning independent coordinate nodes for its partial derivatives.
    with torch.no_grad():
        output = hamiltonian.effective_hamiltonian(r, p, j)
    if not isinstance(output, Tensor) or output.shape != r.shape:
        raise ValueError(
            "hamiltonian must preserve the supplied scalar-intrinsic state shape; "
            f"got effective energy shape {getattr(output, 'shape', None)} for {tuple(r.shape)}"
        )
    return r, p, j, output, torch.is_grad_enabled()


def _evaluate(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> TEOBRadialAccelerationTerms:
    r, p, j, _, track_grad = _prepare_inputs(hamiltonian, radius, pr_star, p_phi)

    with torch.inference_mode(False), torch.enable_grad():
        # Cloning after broadcasting materializes one independent coordinate
        # node per profile sample while retaining any caller-side gradients.
        r_eval = r.clone().requires_grad_(True)
        p_eval = p.clone().requires_grad_(True)
        j_eval = j.clone().requires_grad_(True)
        r0 = r.clone().requires_grad_(True)
        p0 = torch.zeros_like(r0).requires_grad_(True)
        j0 = j.clone().requires_grad_(True)

        a_current, _, xi_current = hamiltonian.metric_potentials(r_eval)
        a0, _, xi0 = hamiltonian.metric_potentials(r0)
        h_current = hamiltonian.effective_hamiltonian(r_eval, p_eval, j_eval)
        h0 = hamiltonian.effective_hamiltonian(r0, p0, j0)
        H_current = hamiltonian.real_hamiltonian_from_eff(h_current)
        H0 = hamiltonian.real_hamiltonian_from_eff(h0)

        current_flow = hamiltonian.equations_of_motion_conservative(
            r_eval, p_eval, j_eval,
        )
        p0_flow = hamiltonian.equations_of_motion_conservative(r0, p0, j0)
        current_velocity, _, current_force = current_flow
        p0_velocity, _, p0_force = p0_flow

        nu = hamiltonian.nu
        z3 = 2.0 * nu * (4.0 - 3.0 * nu)
        q_current = 1.0 + 2.0 * z3 * a_current * p_eval.square() / r_eval.square()
        mobility_current = xi_current * q_current / (H_current * h_current)
        # The p=0 limit is written directly from the Hamilton equation.  It
        # is defined analytically even when the supplied p profile contains 0.
        mobility0 = xi0 / (H0 * h0)

        strict = mobility0 * p0_force
        mixed_current_p0 = mobility_current * p0_force
        mixed_p0_current = mobility0 * current_force
        frozen = mobility_current * current_force

        p0_velocity_radius_partial, p0_velocity_pr_partial = torch.autograd.grad(
            p0_velocity.sum(), (r0, p0), create_graph=True, allow_unused=True,
        )
        p0_velocity_radius_partial = _zero_if_none(p0_velocity_radius_partial, r0)
        p0_velocity_pr_partial = _zero_if_none(p0_velocity_pr_partial, p0)

        values = (
            strict,
            mixed_current_p0,
            mixed_p0_current,
            frozen,
            p0_velocity,
            current_velocity,
            mobility0,
            mobility_current,
            p0_force,
            current_force,
            h0,
            h_current,
            H0,
            H_current,
            p0_velocity_radius_partial,
            p0_velocity_pr_partial,
        )
        for name, value in zip(
            (
                "strict_p0", "current_mobility_p0_force", "p0_mobility_current_force",
                "frozen_mobility", "p0_radial_velocity", "current_radial_velocity",
                "p0_mobility", "current_mobility", "p0_pr_star_rate",
                "current_pr_star_rate", "p0_effective_energy", "current_effective_energy",
                "p0_real_energy", "current_real_energy",
                "p0_radial_velocity_radius_partial", "p0_radial_velocity_pr_star_partial",
            ),
            values,
        ):
            _finite(name, value)
        _positive("p0_effective_energy", h0)
        _positive("current_effective_energy", h_current)
        _positive("p0_real_energy", H0)
        _positive("current_real_energy", H_current)
        _positive("p0_mobility", mobility0)
        _positive("current_mobility", mobility_current)

        result = TEOBRadialAccelerationTerms(*values)

    if track_grad:
        return result
    return TEOBRadialAccelerationTerms(*(
        getattr(result, field.name).detach() for field in fields(result)
    ))


def radial_acceleration_approximations(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> TEOBRadialAccelerationTerms:
    """Return all named zero-spin conservative radial branches.

    ``radius``, ``pr_star`` and ``p_phi`` can be broadcast profile arrays.
    The Hamiltonian's ``nu``, ``chi1`` and ``chi2`` must be scalar tensors;
    intrinsic batch broadcasting is rejected.
    """

    return _evaluate(hamiltonian, radius, pr_star, p_phi)


def strict_leading_p_radial_acceleration(
    hamiltonian: object,
    radius: object,
    p_phi: object,
) -> Tensor:
    """Return the strict ``p_r*=0`` leading-momentum acceleration."""

    r = _reference_tensor(radius, p_phi)
    zeros = torch.zeros_like(_as_real(radius, r, "radius"))
    return radial_acceleration_approximations(hamiltonian, radius, zeros, p_phi).strict_p0


def mixed_current_mobility_radial_acceleration(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> Tensor:
    """Return current mobility times the strict ``p=0`` force."""

    return radial_acceleration_approximations(
        hamiltonian, radius, pr_star, p_phi,
    ).current_mobility_p0_force


def mixed_p0_mobility_radial_acceleration(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> Tensor:
    """Return ``p=0`` mobility times the current conservative force."""

    return radial_acceleration_approximations(
        hamiltonian, radius, pr_star, p_phi,
    ).p0_mobility_current_force


def frozen_mobility_radial_acceleration(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
) -> Tensor:
    """Return the opt in current-state frozen-mobility approximation."""

    return radial_acceleration_approximations(
        hamiltonian, radius, pr_star, p_phi,
    ).frozen_mobility


def approximate_conservative_radial_acceleration(
    hamiltonian: object,
    radius: object,
    pr_star: object,
    p_phi: object,
    *,
    variant: Variant = "strict_p0",
) -> Tensor:
    """Select one explicitly named approximation without changing runtime defaults."""

    terms = radial_acceleration_approximations(hamiltonian, radius, pr_star, p_phi)
    if variant not in {
        "strict_p0", "mixed_current_p0", "mixed_p0_current", "frozen_mobility",
    }:
        raise ValueError(
            "variant must be 'strict_p0', 'mixed_current_p0', "
            "'mixed_p0_current', or 'frozen_mobility'"
        )
    return getattr(terms, variant)


# Equation-facing aliases kept additive and opt in.
conservative_radial_acceleration_p0 = strict_leading_p_radial_acceleration
zero_spin_conservative_radial_acceleration_p0 = strict_leading_p_radial_acceleration
teob_frozen_mobility_radial_acceleration = frozen_mobility_radial_acceleration
zero_spin_frozen_mobility_radial_acceleration = frozen_mobility_radial_acceleration


__all__ = [
    "TEOBRadialAccelerationError",
    "TEOBRadialAccelerationTerms",
    "radial_acceleration_approximations",
    "strict_leading_p_radial_acceleration",
    "mixed_current_mobility_radial_acceleration",
    "mixed_p0_mobility_radial_acceleration",
    "frozen_mobility_radial_acceleration",
    "approximate_conservative_radial_acceleration",
    "conservative_radial_acceleration_p0",
    "zero_spin_conservative_radial_acceleration_p0",
    "teob_frozen_mobility_radial_acceleration",
    "zero_spin_frozen_mobility_radial_acceleration",
]
