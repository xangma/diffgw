"""Clean-room nonspinning TEOB remnant prescriptions.

This module contains the zero-spin reduction of Healy, Lousto and Zlochower,
arXiv:1406.7295v1, Eqs. (14) and (16), with the coefficients printed in that
paper's Table VI.  The spin equation is implicit because its ISCO angular
momentum is evaluated at the remnant spin; a small fixed-point solve is used
for that equation.

Two ISCO-radius conventions are deliberately exposed.  ``"paper-prograde"``
is the signed Bardeen radius used by the published formula.  The TEOB
compatibility convention, ``"teob-compatibility"``, uses ``3 + Z2`` for the
radius while retaining the spin dependence in the Kerr energy and in the
paper's ISCO angular-momentum expression.  At this radius the compatibility
energy is the general circular-orbit Kerr energy; the radius is not claimed to
be a physical prograde ISCO.  No convention is selected by an implicit global
state.

The explicit hybrid helper combines the compatibility-branch Healy final-mass
fraction with the printed Table-I coefficients of Jiménez-Forteza et al.,
arXiv:1611.00332v2, Eq. (7), for the final spin.  It is separate from the
existing UIB2016 helper, whose full-precision standalone behaviour is left
unchanged.

All public functions are real, float64, broadcast-aware, and differentiable
with respect to tensor inputs.  They are intentionally limited to positive
component masses and the nonspinning sector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


# Healy, Lousto & Zlochower, arXiv:1406.7295v1, Table VI.  These are the
# printed decimal inputs, rather than coefficients reconstructed from a later
# implementation or from numerical outputs.
_M0 = 0.951507
_K2D = 1.995246
_K4F = 2.980990
_L0 = 0.686710
_L2D = 0.801838
_L4F = 0.953458

# Jiménez-Forteza et al., arXiv:1611.00332v2, Table I (p. 5), as used in
# Eq. (7).  Keep these printed-table values distinct from the full-precision
# author-supplement constants in teob_uib2016_remnant.py.
_JF_PRINTED_A2 = 3.833
_JF_PRINTED_A3 = -9.49
_JF_PRINTED_A5 = 2.513

PAPER_PROGRADE_ISCO = "paper-prograde"
TEOB_COMPATIBILITY_ISCO = "teob-compatibility"


@dataclass(frozen=True)
class Healy2014RemnantDetails:
    """Values from one zero-spin Healy fixed-point solve.

    ``spin`` is the Healy internal spin used in the mass and spin equations.
    The TEOB hybrid helper deliberately replaces that output spin with the
    printed-table Jiménez-Forteza spin, while retaining ``mass_fraction``.
    """

    mass_fraction: torch.Tensor
    spin: torch.Tensor
    eta: torch.Tensor
    delta_m: torch.Tensor
    isco_radius: torch.Tensor
    isco_energy: torch.Tensor
    isco_angular_momentum: torch.Tensor
    isco_branch: str

    @property
    def healy_spin(self) -> torch.Tensor:
        """Alias making the internal (rather than hybrid) spin explicit."""

        return self.spin


def _coerce_float64(value: torch.Tensor | float, *others: torch.Tensor | float) -> tuple[torch.Tensor, ...]:
    """Convert real values to float64 without detaching tensor gradients."""

    values = (value, *others)
    tensors = [item for item in values if isinstance(item, torch.Tensor)]
    if any(item.is_complex() for item in tensors):
        raise ValueError("TEOB remnant inputs must be real")
    if any(isinstance(item, bool) for item in values) or any(
        item.dtype == torch.bool for item in tensors
    ):
        raise TypeError("TEOB remnant inputs must be real numbers")

    if tensors:
        device = tensors[0].device
        if any(item.device != device for item in tensors):
            raise ValueError("TEOB remnant tensors must share a device")
    else:
        device = torch.device("cpu")

    converted: list[torch.Tensor] = []
    for item in values:
        try:
            converted.append(torch.as_tensor(item, device=device, dtype=torch.float64))
        except (TypeError, RuntimeError) as exc:
            raise ValueError("TEOB remnant inputs must be finite real numbers") from exc
    return tuple(converted)


def _validate_finite(values: tuple[torch.Tensor, ...], message: str) -> None:
    if any(torch.any(~torch.isfinite(item)) for item in values):
        raise ValueError(message)


def _component_parameters(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return broadcasted ``(m1, m2, eta, delta_m)`` with stable scaling."""

    m1, m2 = _coerce_float64(mass1, mass2)
    m1, m2 = torch.broadcast_tensors(m1, m2)
    _validate_finite((m1, m2), "component masses must be finite")
    if torch.any((m1 <= 0) | (m2 <= 0)):
        raise ValueError("component masses must be positive")

    # Scaling by the larger component prevents overflow for dimensional
    # inputs while preserving both eta and the signed mass difference.
    scale = torch.maximum(m1, m2)
    x1, x2 = m1 / scale, m2 / scale
    denominator = x1 + x2
    eta = (x1 * x2) / denominator.square()
    eta = torch.minimum(eta, eta.new_tensor(0.25))
    delta_m = (x1 - x2) / denominator
    return m1, m2, eta, delta_m


def _validate_eta(eta: torch.Tensor | float) -> torch.Tensor:
    (eta,) = _coerce_float64(eta)
    _validate_finite((eta,), "symmetric mass ratio eta must be finite")
    if torch.any((eta < 0) | (eta > 0.25)):
        raise ValueError("symmetric mass ratio eta must satisfy 0 <= eta <= 1/4")
    return eta


def _eta_delta_from_q(q: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    (q,) = _coerce_float64(q)
    _validate_finite((q,), "mass ratio q must be finite")
    if torch.any(q < 1):
        raise ValueError("mass ratio q must satisfy q >= 1")
    # x=1/q avoids overflow in the usual q/(1+q)^2 expression.
    x = 1.0 / q
    return x / (1.0 + x).square(), (x - 1.0) / (x + 1.0)


def _normalise_branch(isco_branch: str) -> str:
    if not isinstance(isco_branch, str):
        raise TypeError("isco_branch must be a string")
    key = isco_branch.strip().lower().replace("_", "-")
    if key in {"paper", "prograde", "paper-prograde", "published-prograde"}:
        return PAPER_PROGRADE_ISCO
    if key in {
        "compatibility",
        "signless",
        "teob-compatibility",
        "teob-signless-radius",
    }:
        return TEOB_COMPATIBILITY_ISCO
    raise ValueError(
        "unknown isco_branch; choose 'paper-prograde' or 'teob-compatibility'"
    )


def _bardeen_z1_z2(spin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the standard Bardeen ``(Z1,Z2)`` functions."""

    # The solve below stays strictly inside the Kerr range.  Clamping only
    # removes an ulp-sized negative radicand at a boundary and does not make a
    # non-Kerr input valid.
    if torch.any((spin < -1) | (spin > 1)) or torch.any(~torch.isfinite(spin)):
        raise ValueError("Kerr spin must satisfy -1 <= spin <= 1")
    one_minus_spin2 = torch.clamp(1.0 - spin.square(), min=0.0)
    z1 = 1.0 + one_minus_spin2.pow(1.0 / 3.0) * (
        torch.clamp(1.0 + spin, min=0.0).pow(1.0 / 3.0)
        + torch.clamp(1.0 - spin, min=0.0).pow(1.0 / 3.0)
    )
    z2 = torch.sqrt(torch.clamp(3.0 * spin.square() + z1.square(), min=0.0))
    return z1, z2


def healy2014_isco_radius(
    spin: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
) -> torch.Tensor:
    """Return the dimensionless Kerr radius for a named remnant convention.

    The paper branch is ``3+Z2-sgn(a)*sqrt((3-Z1)*(3+Z1+2*Z2))``.  The TEOB
    compatibility branch intentionally omits the signed square-root term and
    is therefore ``3+Z2``; it is a compatibility radius rather than the
    physical prograde ISCO.
    """

    (spin,) = _coerce_float64(spin)
    branch = _normalise_branch(isco_branch)
    z1, z2 = _bardeen_z1_z2(spin)
    if branch == TEOB_COMPATIBILITY_ISCO:
        return 3.0 + z2

    radicand = torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=0.0)
    # The signed square root is algebraically smooth at a=0 but its direct
    # autograd expression is 0*sqrt'(0).  A local polynomial has the exact
    # first derivative and keeps the public helper finite at Schwarzschild.
    # Keep the inactive branch's derivative finite as well: torch.where still
    # backpropagates through both operands, and sqrt'(0) would otherwise turn
    # the zero signed factor into a NaN at exactly Schwarzschild.
    signed_root = torch.sign(spin) * torch.sqrt(radicand + spin.new_tensor(1.0e-30))
    radius = 3.0 + z2 - signed_root
    near_zero = spin.abs() < 1.0e-7
    radius_local = 6.0 - 4.0 * torch.sqrt(spin.new_tensor(2.0 / 3.0)) * spin
    return torch.where(near_zero, radius_local, radius)


def healy2014_specific_isco_energy(
    radius: torch.Tensor | float,
    spin: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
) -> torch.Tensor:
    """Return the specific Kerr circular energy used by the mass equation.

    At the paper's physical ISCO this is algebraically the familiar
    ``sqrt(1-2/(3*r_ISCO))``.  The general ``E(r,a)`` form is retained so the
    TEOB compatibility branch keeps its explicit dependence on the iterated
    spin at ``r=3+Z2``.
    """

    radius, spin = _coerce_float64(radius, spin)
    radius, spin = torch.broadcast_tensors(radius, spin)
    _normalise_branch(isco_branch)
    _validate_finite((radius, spin), "Kerr radius and spin must be finite")
    if torch.any(radius <= 0) or torch.any((spin < -1) | (spin > 1)):
        raise ValueError("Kerr radius must be positive and spin must be in [-1, 1]")
    sqrt_r = torch.sqrt(radius)
    radial_factor = radius * sqrt_r - 3.0 * sqrt_r + 2.0 * spin
    if torch.any(radial_factor <= 0):
        raise ValueError("Kerr circular energy is undefined at the supplied radius")
    return (radius * sqrt_r - 2.0 * sqrt_r + spin) / (
        radius.pow(0.75) * torch.sqrt(radial_factor)
    )


def healy2014_specific_isco_angular_momentum(
    radius: torch.Tensor | float,
    spin: torch.Tensor | float,
) -> torch.Tensor:
    """Return the paper's dimensionless ``J_ISCO(r,a)`` expression.

    This is Eq. (2.8) in the Kerr-ISCO convention used by Healy et al. and
    Jiménez-Forteza et al.  On the compatibility radius it is deliberately
    evaluated at ``3+Z2`` even though that radius is not a physical ISCO.
    """

    radius, spin = _coerce_float64(radius, spin)
    radius, spin = torch.broadcast_tensors(radius, spin)
    _validate_finite((radius, spin), "Kerr radius and spin must be finite")
    if torch.any(radius <= 0) or torch.any((spin < -1) | (spin > 1)):
        raise ValueError("Kerr radius must be positive and spin must be in [-1, 1]")
    return 2.0 * (3.0 * torch.sqrt(radius) - 2.0 * spin) / torch.sqrt(3.0 * radius)


def jimenez_forteza_printed_table_nonspinning_final_spin(
    eta: torch.Tensor | float,
) -> torch.Tensor:
    """Return the printed-Table-I Jiménez-Forteza zero-spin final spin.

    The coefficients are ``a2=3.833``, ``a3=-9.49`` and ``a5=2.513`` from
    Table I of arXiv:1611.00332v2, inserted in Eq. (7).  This function is
    intentionally separate from the full-precision standalone UIB2016 fit.
    """

    eta = _validate_eta(eta)
    numerator = (
        2.0 * torch.sqrt(eta.new_tensor(3.0)) * eta
        + eta.new_tensor(5.24 * _JF_PRINTED_A2) * eta.square()
        + eta.new_tensor(1.3 * _JF_PRINTED_A3) * eta.pow(3)
    )
    denominator = 1.0 + eta.new_tensor(2.88 * _JF_PRINTED_A5) * eta
    return numerator / denominator


# Descriptive aliases make the model-version distinction obvious to callers
# while retaining one implementation and one source of the printed constants.
uib2016_printed_table_nonspinning_final_spin = (
    jimenez_forteza_printed_table_nonspinning_final_spin
)
jimenez_forteza_2017_printed_table_nonspinning_final_spin = (
    jimenez_forteza_printed_table_nonspinning_final_spin
)


def _solve_healy_spin(
    eta: torch.Tensor,
    delta_m: torch.Tensor,
    *,
    isco_branch: str,
    tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not torch.isfinite(torch.as_tensor(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")

    # Zero-spin reduction of Healy et al. Eq. (16), Table VI.
    spin = (4.0 * eta).square() * (
        eta.new_tensor(_L0)
        + eta.new_tensor(_L2D) * delta_m.square()
        + eta.new_tensor(_L4F) * delta_m.pow(4)
    )
    coefficient = eta * delta_m.pow(6)
    branch = _normalise_branch(isco_branch)

    # The iteration is expressed with out-of-place tensors to preserve the
    # autograd path through the converged published equation.
    for _ in range(max_iterations):
        radius = healy2014_isco_radius(spin, isco_branch=branch)
        angular_momentum = healy2014_specific_isco_angular_momentum(radius, spin)
        new_spin = (4.0 * eta).square() * (
            eta.new_tensor(_L0)
            + eta.new_tensor(_L2D) * delta_m.square()
            + eta.new_tensor(_L4F) * delta_m.pow(4)
        ) + coefficient * angular_momentum
        difference = torch.max(torch.abs(new_spin - spin)).item()
        scale = torch.max(torch.abs(new_spin)).item()
        spin = new_spin
        if difference <= float(tolerance) * max(1.0, scale):
            break
    else:
        raise RuntimeError(
            f"Healy remnant spin fixed-point solve did not converge in {max_iterations} iterations"
        )

    if torch.any(~torch.isfinite(spin)) or torch.any(spin.abs() > 1.0):
        raise RuntimeError("Healy remnant spin fixed-point solve left the Kerr domain")
    return spin


def _healy_details_from_parameters(
    eta: torch.Tensor,
    delta_m: torch.Tensor,
    *,
    isco_branch: str,
    tolerance: float,
    max_iterations: int,
) -> Healy2014RemnantDetails:
    branch = _normalise_branch(isco_branch)
    spin = _solve_healy_spin(
        eta,
        delta_m,
        isco_branch=branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    radius = healy2014_isco_radius(spin, isco_branch=branch)
    energy = healy2014_specific_isco_energy(radius, spin, isco_branch=branch)
    angular_momentum = healy2014_specific_isco_angular_momentum(radius, spin)

    # Zero-spin reduction of Healy et al. Eq. (14), Table VI.
    mass_fraction = (4.0 * eta).square() * (
        eta.new_tensor(_M0)
        + eta.new_tensor(_K2D) * delta_m.square()
        + eta.new_tensor(_K4F) * delta_m.pow(4)
    ) + (1.0 + eta * (energy + 11.0)) * delta_m.pow(6)
    return Healy2014RemnantDetails(
        mass_fraction=mass_fraction,
        spin=spin,
        eta=eta,
        delta_m=delta_m,
        isco_radius=radius,
        isco_energy=energy,
        isco_angular_momentum=angular_momentum,
        isco_branch=branch,
    )


def healy2014_nonspinning_remnant_details(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> Healy2014RemnantDetails:
    """Return all intermediate values of the zero-spin Healy prescription."""

    _, _, eta, delta_m = _component_parameters(mass1, mass2)
    return _healy_details_from_parameters(
        eta,
        delta_m,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def healy2014_nonspinning_final_mass_fraction(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> torch.Tensor:
    """Return ``M_f/(m1+m2)`` from Healy et al. Eq. (14)."""

    return healy2014_nonspinning_remnant_details(
        mass1,
        mass2,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).mass_fraction


def healy2014_nonspinning_final_spin(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> torch.Tensor:
    """Return the implicit zero-spin Healy final spin from Eq. (16)."""

    return healy2014_nonspinning_remnant_details(
        mass1,
        mass2,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).spin


def healy2014_nonspinning_remnant(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(M_f/(m1+m2), chi_f)`` for positive nonspinning masses."""

    details = healy2014_nonspinning_remnant_details(
        mass1,
        mass2,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return details.mass_fraction, details.spin


def healy2014_nonspinning_remnant_from_q(
    q: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the remnant tuple for ``q=m_large/m_small >= 1``."""

    eta, delta_m = _eta_delta_from_q(q)
    details = _healy_details_from_parameters(
        eta,
        delta_m,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return details.mass_fraction, details.spin


def healy2014_nonspinning_final_mass_fraction_from_q(
    q: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> torch.Tensor:
    """Return the Healy mass fraction for ``q=m_large/m_small >= 1``."""

    return healy2014_nonspinning_remnant_from_q(
        q,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )[0]


def healy2014_nonspinning_final_spin_from_q(
    q: torch.Tensor | float,
    *,
    isco_branch: str = PAPER_PROGRADE_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> torch.Tensor:
    """Return the Healy implicit spin for ``q=m_large/m_small >= 1``."""

    return healy2014_nonspinning_remnant_from_q(
        q,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )[1]


def teob_hybrid_nonspinning_remnant(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
    *,
    isco_branch: str = TEOB_COMPATIBILITY_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return TEOB's explicit Healy-mass/printed-Jiménez-spin hybrid.

    The compatibility branch is the signless ``3+Z2`` radius.  The branch is
    retained as a keyword for provenance, but selecting the paper branch is
    rejected because it would no longer be the TEOB hybrid prescription.
    """

    if _normalise_branch(isco_branch) != TEOB_COMPATIBILITY_ISCO:
        raise ValueError(
            "teob_hybrid_nonspinning_remnant requires the 'teob-compatibility' branch"
        )
    details = healy2014_nonspinning_remnant_details(
        mass1,
        mass2,
        isco_branch=isco_branch,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return details.mass_fraction, jimenez_forteza_printed_table_nonspinning_final_spin(
        details.eta
    )


def teob_hybrid_nonspinning_remnant_from_q(
    q: torch.Tensor | float,
    *,
    isco_branch: str = TEOB_COMPATIBILITY_ISCO,
    tolerance: float = 1.0e-14,
    max_iterations: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the explicit TEOB hybrid for ``q=m_large/m_small >= 1``."""

    if _normalise_branch(isco_branch) != TEOB_COMPATIBILITY_ISCO:
        raise ValueError(
            "teob_hybrid_nonspinning_remnant requires the 'teob-compatibility' branch"
        )
    eta, delta_m = _eta_delta_from_q(q)
    details = _healy_details_from_parameters(
        eta,
        delta_m,
        isco_branch=TEOB_COMPATIBILITY_ISCO,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return details.mass_fraction, jimenez_forteza_printed_table_nonspinning_final_spin(
        details.eta
    )


teob_healy2014_uib2016_nonspinning_remnant = teob_hybrid_nonspinning_remnant


__all__ = [
    "Healy2014RemnantDetails",
    "PAPER_PROGRADE_ISCO",
    "TEOB_COMPATIBILITY_ISCO",
    "healy2014_isco_radius",
    "healy2014_specific_isco_energy",
    "healy2014_specific_isco_angular_momentum",
    "jimenez_forteza_printed_table_nonspinning_final_spin",
    "jimenez_forteza_2017_printed_table_nonspinning_final_spin",
    "uib2016_printed_table_nonspinning_final_spin",
    "healy2014_nonspinning_remnant_details",
    "healy2014_nonspinning_final_mass_fraction",
    "healy2014_nonspinning_final_spin",
    "healy2014_nonspinning_remnant",
    "healy2014_nonspinning_remnant_from_q",
    "healy2014_nonspinning_final_mass_fraction_from_q",
    "healy2014_nonspinning_final_spin_from_q",
    "teob_hybrid_nonspinning_remnant",
    "teob_hybrid_nonspinning_remnant_from_q",
    "teob_healy2014_uib2016_nonspinning_remnant",
]
