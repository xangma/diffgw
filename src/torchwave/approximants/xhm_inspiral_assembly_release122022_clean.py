# SPDX-License-Identifier: MIT
"""Clean XHM inspiral amplitude assembly for release ``122022``.

This module is a small, source-derived adapter for the current XHM amplitude
contract.  It deliberately contains no parameter-space fit values and never
calls the inherited fit provider.  A caller supplies the three *geometric
Fourier strain amplitudes* at the ascending nodes
``(.5, .75, 1.0) * f_ins`` together with an explicit provenance record.

For a mode ``(ell, m)`` the current release uses

``f_ins = (m / 2) * f_MECO22``

and the Appendix-E re-expanded polynomial for every supported mode, including
``(2, 1)``.  The named compatibility signs for ``(3, 2)`` and ``(4, 4)`` are
retained through :func:`xhm_amplitude_clean.hhat_lm_compatibility`.

Let ``H_PN = |hhat_lm| * (m/2)**(7/6)`` and

``K = pi*sqrt(2*eta/3) * (pi*M_seconds)**(-7/6)``.

The geometric amplitude is assembled as

``A_geo(f) = K*f**(-7/6) * (H_PN(f) + sum(c_k*u**p_k))``

with ``u=f/f_ins`` and ``p=(7/3, 8/3, 3)``.  The coefficients are obtained
from the supplied geometric node amplitudes after dividing each PN residual
by its corresponding ``K*f_i**(-7/6)``.  This is the same algebra as solving
the physical equation directly; the common scale cancels from the linear
system.  ``physical_inspiral_amplitude`` applies the additional ``M**2/D``
factor used by the frequency-domain strain convention.

Only ``q < 20`` and modes ``(2,1)``, ``(3,2)``, ``(3,3)``, and ``(4,4)`` are
handled.  There are no automatic node vetoes in this adapter: the retained
node tuple is an explicit input (defaulting to all three published nodes),
and omitted or unsupported model policies are rejected rather than inferred.
The implementation is intentionally independent of the canonical XHM runtime;
passing public or runtime oracle values as calibration data would violate its
provenance contract.

Primary equation provenance: García-Quirós et al., arXiv:2001.10914v2,
Sec. IV.A.1 and Appendix E.  The current-release normalization and node-unit
choice follow the sanitized local XHM contract; they are not a parity claim.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .xhm_amplitude_clean import hhat_lm_compatibility, hhat_lm_printed


__all__ = [
    "RELEASE_VERSION",
    "SUPPORTED_MODES",
    "COLLOCATION_FRACTIONS",
    "PSEUDO_PN_POWERS",
    "DEFAULT_RETAINED_NODES",
    "CALIBRATION_QUANTITY",
    "CalibrationProvenance",
    "validate_calibration_provenance",
    "inspiral_cut_frequency",
    "f_ins_from_meco22",
    "pn_normalized_amplitude",
    "pn_geometric_amplitude",
    "normalized_inspiral_amplitude",
    "geometric_inspiral_amplitude",
    "physical_inspiral_amplitude",
    "current_release_inspiral_amplitude",
]


RELEASE_VERSION = 122022
SUPPORTED_MODES = ((2, 1), (3, 2), (3, 3), (4, 4))
COLLOCATION_FRACTIONS = (0.5, 0.75, 1.0)
PSEUDO_PN_POWERS = (7.0 / 3.0, 8.0 / 3.0, 3.0)
DEFAULT_RETAINED_NODES = (0, 1, 2)
CALIBRATION_QUANTITY = "geometric_fourier_strain_amplitude"


@dataclass(frozen=True)
class CalibrationProvenance:
    """Required metadata for caller-supplied current-release node values.

    ``source`` should identify the independently supplied calibration asset.
    The adapter records no values and does not inspect or obtain them.  A
    mapping with the same four fields is accepted by
    :func:`validate_calibration_provenance` for serialization-friendly APIs.
    """

    source: str
    release: int = RELEASE_VERSION
    quantity: str = CALIBRATION_QUANTITY
    node_fractions: tuple[float, float, float] = COLLOCATION_FRACTIONS


def _finite_real(value: object, name: str) -> torch.Tensor:
    """Convert a real input to float64 while preserving autograd edges."""

    raw = torch.as_tensor(value)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise TypeError(f"{name} must be finite real values")
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().detach().cpu()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_mode(mode: object) -> tuple[int, int]:
    try:
        candidate = tuple(mode)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("mode must be a two-integer (ell, m) tuple") from exc
    if len(candidate) != 2 or any(
        isinstance(part, bool) or not isinstance(part, int) for part in candidate
    ):
        raise TypeError("mode must be a two-integer (ell, m) tuple")
    selected = (candidate[0], candidate[1])
    if selected not in SUPPORTED_MODES:
        raise ValueError(
            f"release {RELEASE_VERSION} supports only modes {SUPPORTED_MODES}; "
            f"received {selected}"
        )
    return selected


def _validate_retained_nodes(retained_nodes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(retained_nodes, (str, bytes)):
        raise TypeError("retained_nodes must be a tuple of node indices")
    try:
        selected = tuple(retained_nodes)
    except TypeError as exc:
        raise TypeError("retained_nodes must be a tuple of node indices") from exc
    if any(
        isinstance(node, bool) or not isinstance(node, int) for node in selected
    ):
        raise TypeError("retained_nodes entries must be integers")
    if len(selected) > 3:
        raise ValueError("at most the three published collocation nodes may be retained")
    if any(node not in (0, 1, 2) for node in selected):
        raise ValueError("retained_nodes entries must be 0, 1, or 2")
    if len(set(selected)) != len(selected):
        raise ValueError("retained_nodes must contain distinct node indices")
    # The node values themselves have a fixed ascending trailing order.  The
    # retained tuple is an explicit row selection, so callers may name those
    # rows in any order; the linear solve is invariant under that permutation.
    return selected


def validate_calibration_provenance(
    provenance: CalibrationProvenance | Mapping[str, Any],
) -> CalibrationProvenance:
    """Validate and normalize explicit provenance for node amplitudes.

    The release and quantity checks prevent accidentally passing the older
    normalized-H calibration contract into this geometric-amplitude adapter.
    The source label is metadata only; no source is opened or evaluated here.
    """

    if isinstance(provenance, CalibrationProvenance):
        record = provenance
    elif isinstance(provenance, Mapping):
        required = ("source", "release", "quantity", "node_fractions")
        missing = [key for key in required if key not in provenance]
        if missing:
            raise ValueError(
                "calibration_provenance is missing required fields: "
                + ", ".join(missing)
            )
        try:
            fractions = tuple(float(item) for item in provenance["node_fractions"])
        except (TypeError, ValueError) as exc:
            raise ValueError("node_fractions must contain three finite numbers") from exc
        if len(fractions) != 3 or not all(math.isfinite(item) for item in fractions):
            raise ValueError("node_fractions must contain three finite numbers")
        try:
            record = CalibrationProvenance(
                source=provenance["source"],
                release=provenance["release"],
                quantity=provenance["quantity"],
                node_fractions=fractions,  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise TypeError("calibration provenance fields have invalid types") from exc
    else:
        raise TypeError(
            "calibration_provenance must be CalibrationProvenance or a mapping"
        )

    if not isinstance(record.source, str) or not record.source.strip():
        raise ValueError("calibration_provenance.source must be a non-empty label")
    if isinstance(record.release, bool) or record.release != RELEASE_VERSION:
        raise ValueError(
            f"calibration provenance must identify release {RELEASE_VERSION}"
        )
    if record.quantity != CALIBRATION_QUANTITY:
        raise ValueError(
            "calibration provenance quantity must be "
            f"{CALIBRATION_QUANTITY!r}"
        )
    try:
        fractions = tuple(float(item) for item in record.node_fractions)
    except (TypeError, ValueError) as exc:
        raise ValueError("node_fractions must contain three finite numbers") from exc
    if len(fractions) != 3 or not all(math.isfinite(item) for item in fractions):
        raise ValueError("node_fractions must contain three finite numbers")
    if any(
        not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-14)
        for got, expected in zip(fractions, COLLOCATION_FRACTIONS)
    ):
        raise ValueError(
            "calibration provenance node_fractions must be "
            f"{COLLOCATION_FRACTIONS} in ascending order"
        )
    return CalibrationProvenance(
        source=record.source,
        release=RELEASE_VERSION,
        quantity=CALIBRATION_QUANTITY,
        node_fractions=COLLOCATION_FRACTIONS,
    )


def inspiral_cut_frequency(mode: object, f_meco_22: object) -> torch.Tensor:
    """Return current-release ``f_ins(l,m)=(m/2)*f_MECO22``.

    The MECO value may be scalar or batched and may carry gradients.  No ISCO,
    ringdown, spin, or inherited cut prescription is consulted.
    """

    selected = _validate_mode(mode)
    meco = _finite_real(f_meco_22, "f_meco_22")
    if bool((meco <= 0).any()):
        raise ValueError("f_meco_22 must be positive")
    return (selected[1] / 2.0) * meco


f_ins_from_meco22 = inspiral_cut_frequency


def _validate_intrinsics(
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    f_ins: object | None = None,
) -> tuple[torch.Tensor, ...]:
    values = [_finite_real(value, name) for value, name in zip(
        (frequency, total_mass_seconds, m1, m2, chi1, chi2),
        ("frequency", "total_mass_seconds", "m1", "m2", "chi1", "chi2"),
    )]
    if f_ins is not None:
        values.append(_finite_real(f_ins, "f_ins"))
    broadcast = torch.broadcast_tensors(*values)
    f, mt, mass1, mass2, spin1, spin2 = broadcast[:6]
    if bool((f <= 0).any()):
        raise ValueError("frequency must be positive")
    if bool((mt <= 0).any()):
        raise ValueError("total_mass_seconds must be positive")
    if bool((mass1 <= 0).any()) or bool((mass2 <= 0).any()):
        raise ValueError("component masses must be positive")
    if bool((spin1.abs() > 1).any()) or bool((spin2.abs() > 1).any()):
        raise ValueError("component spins must lie in [-1, 1]")
    q = torch.maximum(mass1, mass2) / torch.minimum(mass1, mass2)
    if bool((q >= 20).any()):
        raise ValueError("release 122022 clean assembly requires q < 20")
    if f_ins is None:
        return f, mt, mass1, mass2, spin1, spin2, q
    cut = broadcast[6]
    if bool((cut <= 0).any()):
        raise ValueError("f_ins must be positive")
    if bool((f > cut).any()):
        raise ValueError("assembly frequencies must satisfy frequency <= f_ins")
    return f, cut, mt, mass1, mass2, spin1, spin2, q


def _node_values(value: object) -> torch.Tensor:
    raw = torch.as_tensor(value)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise TypeError("collocation_values must be finite real amplitudes")
    values = torch.as_tensor(value, dtype=torch.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(
            "collocation_values must have exactly three trailing node values "
            "in ascending (.5,.75,1) order"
        )
    if not bool(torch.isfinite(values).all().detach().cpu()):
        raise ValueError("collocation_values must be finite")
    if bool((values < 0).any()):
        raise ValueError("geometric Fourier strain amplitudes must be non-negative")
    return values


def _expand_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return value.expand(shape)


def _pn_normalized_from_broadcast(
    mode: tuple[int, int],
    frequency: torch.Tensor,
    total_mass_seconds: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
    *,
    compatibility: bool,
) -> torch.Tensor:
    total = m1 + m2
    eta = m1 * m2 / total.square()
    delta = (m1 - m2) / total
    chi_s = (chi1 + chi2) / 2.0
    chi_a = (chi1 - chi2) / 2.0
    v = (2.0 * math.pi * frequency * total_mass_seconds / mode[1]).pow(1.0 / 3.0)
    if compatibility and mode in ((3, 2), (4, 4)):
        hhat = hhat_lm_compatibility(mode, v, eta, delta, chi_s, chi_a)
    else:
        # The current release uses the re-expanded Appendix-E polynomial for
        # 21 as well as 33.  ``compatibility=False`` is a named paper control
        # for the retained printed 32/44 signs; it never selects xhm_spa_clean.
        hhat = hhat_lm_printed(mode, v, eta, delta, chi_s, chi_a)
    return hhat.abs() * (mode[1] / 2.0) ** (7.0 / 6.0)


def pn_normalized_amplitude(
    mode: object,
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    compatibility: bool = True,
) -> torch.Tensor:
    """Return the re-expanded Appendix-E ``H_PN`` amplitude.

    This low-level baseline has no calibration nodes.  It remains bounded to
    the current release's ``q<20`` domain and uses no non-reexpanded 21 SPA
    helper.
    """

    selected = _validate_mode(mode)
    if not isinstance(compatibility, bool):
        raise TypeError("compatibility must be a boolean")
    f, mt, mass1, mass2, spin1, spin2, _ = _validate_intrinsics(
        frequency, total_mass_seconds, m1, m2, chi1, chi2
    )
    return _pn_normalized_from_broadcast(
        selected, f, mt, mass1, mass2, spin1, spin2,
        compatibility=compatibility,
    )


def _geometric_scale(
    frequency: torch.Tensor,
    total_mass_seconds: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
) -> torch.Tensor:
    total = m1 + m2
    eta = m1 * m2 / total.square()
    # K*f**(-7/6), written as one product to avoid introducing a separate
    # dimensional convention.  The equivalent common K is
    # pi*sqrt(2*eta/3)*(pi*M_seconds)**(-7/6).
    return (
        math.pi
        * torch.sqrt(2.0 * eta / 3.0)
        * (math.pi * frequency * total_mass_seconds).pow(-7.0 / 6.0)
    )


def pn_geometric_amplitude(
    mode: object,
    frequency: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    compatibility: bool = True,
) -> torch.Tensor:
    """Return the geometric Fourier strain amplitude of the PN baseline."""

    selected = _validate_mode(mode)
    if not isinstance(compatibility, bool):
        raise TypeError("compatibility must be a boolean")
    f, mt, mass1, mass2, spin1, spin2, _ = _validate_intrinsics(
        frequency, total_mass_seconds, m1, m2, chi1, chi2
    )
    hpn = _pn_normalized_from_broadcast(
        selected, f, mt, mass1, mass2, spin1, spin2,
        compatibility=compatibility,
    )
    return _geometric_scale(f, mt, mass1, mass2) * hpn


def _assemble(
    mode: object,
    frequency: object,
    f_ins: object,
    collocation_values: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    retained_nodes: Sequence[int],
    calibration_provenance: CalibrationProvenance | Mapping[str, Any],
    compatibility: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    selected = _validate_mode(mode)
    if not isinstance(compatibility, bool):
        raise TypeError("compatibility must be a boolean")
    retained = _validate_retained_nodes(retained_nodes)
    validate_calibration_provenance(calibration_provenance)
    node_values = _node_values(collocation_values)
    f, cut, mt, mass1, mass2, spin1, spin2, q = _validate_intrinsics(
        frequency, total_mass_seconds, m1, m2, chi1, chi2, f_ins=f_ins
    )
    del q  # q is returned by validation only to make the domain check explicit.

    target = torch.broadcast_shapes(f.shape, node_values.shape[:-1])
    f = _expand_to_shape(f, target)
    cut = _expand_to_shape(cut, target)
    mt = _expand_to_shape(mt, target)
    mass1 = _expand_to_shape(mass1, target)
    mass2 = _expand_to_shape(mass2, target)
    spin1 = _expand_to_shape(spin1, target)
    spin2 = _expand_to_shape(spin2, target)
    node_values = _expand_to_shape(node_values, target + (3,))

    fractions = torch.as_tensor(
        COLLOCATION_FRACTIONS, dtype=torch.float64, device=cut.device
    )
    node_frequency = cut.unsqueeze(-1) * fractions
    node_pn = _pn_normalized_from_broadcast(
        selected,
        node_frequency,
        mt.unsqueeze(-1),
        mass1.unsqueeze(-1),
        mass2.unsqueeze(-1),
        spin1.unsqueeze(-1),
        spin2.unsqueeze(-1),
        compatibility=compatibility,
    )
    node_scale = _geometric_scale(
        node_frequency,
        mt.unsqueeze(-1),
        mass1.unsqueeze(-1),
        mass2.unsqueeze(-1),
    )

    active = len(retained)
    if active == 0:
        coefficients = torch.zeros(
            target + (3,), dtype=torch.float64, device=cut.device
        )
    else:
        powers = torch.as_tensor(
            PSEUDO_PN_POWERS[:active], dtype=torch.float64, device=cut.device
        )
        rows = torch.as_tensor(retained, dtype=torch.long, device=cut.device)
        design = fractions.index_select(0, rows).unsqueeze(-1).pow(powers)
        residual = (node_values - node_scale * node_pn).index_select(-1, rows)
        # Divide by the common K*f_i**(-7/6) before solving.  This is
        # algebraically identical to solving in physical/geometric units and
        # keeps the small geometric scales out of the linear system.
        rhs = (residual / node_scale.index_select(-1, rows)).unsqueeze(-1)
        solved = torch.linalg.solve(design, rhs)[..., 0]
        tail = torch.zeros(
            target + (3 - active,), dtype=solved.dtype, device=solved.device
        )
        coefficients = torch.cat((solved, tail), dim=-1)

    u = f / cut
    powers_all = torch.as_tensor(
        PSEUDO_PN_POWERS, dtype=torch.float64, device=cut.device
    )
    basis = u.unsqueeze(-1).pow(powers_all)
    hpn = _pn_normalized_from_broadcast(
        selected, f, mt, mass1, mass2, spin1, spin2,
        compatibility=compatibility,
    )
    normalized = hpn + (coefficients * basis).sum(dim=-1)
    geometric = _geometric_scale(f, mt, mass1, mass2) * normalized
    prepared = (f, cut, mt, mass1, mass2, spin1, spin2)
    return normalized, geometric, prepared


def normalized_inspiral_amplitude(
    mode: object,
    frequency: object,
    f_ins: object,
    collocation_values: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    retained_nodes: Sequence[int] = DEFAULT_RETAINED_NODES,
    calibration_provenance: CalibrationProvenance | Mapping[str, Any],
    compatibility: bool = True,
) -> torch.Tensor:
    """Return dimensionless ``H`` assembled from geometric node amplitudes."""

    normalized, _, _ = _assemble(
        mode, frequency, f_ins, collocation_values,
        total_mass_seconds, m1, m2, chi1, chi2,
        retained_nodes=retained_nodes,
        calibration_provenance=calibration_provenance,
        compatibility=compatibility,
    )
    return normalized


def geometric_inspiral_amplitude(
    mode: object,
    frequency: object,
    f_ins: object,
    collocation_values: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    retained_nodes: Sequence[int] = DEFAULT_RETAINED_NODES,
    calibration_provenance: CalibrationProvenance | Mapping[str, Any],
    compatibility: bool = True,
) -> torch.Tensor:
    """Return the assembled absolute geometric Fourier strain amplitude."""

    _, geometric, _ = _assemble(
        mode, frequency, f_ins, collocation_values,
        total_mass_seconds, m1, m2, chi1, chi2,
        retained_nodes=retained_nodes,
        calibration_provenance=calibration_provenance,
        compatibility=compatibility,
    )
    return geometric


def physical_inspiral_amplitude(
    mode: object,
    frequency: object,
    f_ins: object,
    collocation_values: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    distance_seconds: object,
    *,
    retained_nodes: Sequence[int] = DEFAULT_RETAINED_NODES,
    calibration_provenance: CalibrationProvenance | Mapping[str, Any],
    compatibility: bool = True,
) -> torch.Tensor:
    """Return the physical Fourier-domain amplitude ``(M**2/D)*A_geo``."""

    _, geometric, prepared = _assemble(
        mode, frequency, f_ins, collocation_values,
        total_mass_seconds, m1, m2, chi1, chi2,
        retained_nodes=retained_nodes,
        calibration_provenance=calibration_provenance,
        compatibility=compatibility,
    )
    distance = _finite_real(distance_seconds, "distance_seconds")
    if bool((distance <= 0).any()):
        raise ValueError("distance_seconds must be positive")
    mt = prepared[2]
    target = torch.broadcast_shapes(geometric.shape, distance.shape)
    geometric = geometric.expand(target)
    mt = mt.expand(target)
    distance = distance.expand(target)
    return mt.square() / distance * geometric


current_release_inspiral_amplitude = geometric_inspiral_amplitude
