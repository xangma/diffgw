# SPDX-License-Identifier: MIT
"""Clean-room inspiral amplitude assembly for the published XHM ansatz.

The source is arXiv:2001.10914v2, Sec. IV A.1 (the unnumbered amplitude
ansatz following Eq. (``insp_amp``)) and Appendix E.  The reconstructed,
dimensionless quantity is

``H_lm = |A_lm^PN| / A22^0 + alpha*u**(7/3) + beta*u**(8/3) + gamma*u**3``

with ``u=f/f_ins`` and collocation nodes ``u=(.5,.75,1)``.  The caller
supplies the three collocation values and the cut frequency; this module does
not contain a calibration table or import an inherited XHM implementation.

``normalized_inspiral_amplitude`` accepts collocation values in the paper's
dimensionless ``H_lm`` normalization.  ``physical_inspiral_amplitude`` applies
the common ``M**2/D`` Fourier-domain prefactor and therefore retains absolute
scale.  Both functions return amplitude magnitudes and are defined only for
the bounded modes ``(2,1)``, ``(3,2)``, ``(3,3)``, and ``(4,4)`` with ``q<20``.  The
``(2,1)`` PN baseline uses the non-reexpanded SPA product from Appendix E,
which is valid on the wider published ``q<40`` branch; the outer domain here
is intentionally narrower.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .xhm_amplitude_clean import (
    collocation_coefficients,
    normalized_pn_amplitude,
    normalized_pn_amplitude_compatibility,
)
from .xhm_spa_clean import normalized_spa_amplitude_21


__all__ = [
    "SUPPORTED_MODES",
    "COLLOCATION_FRACTIONS",
    "PRINTED_CUT_VERSION",
    "COMPATIBILITY_CUT_VERSION",
    "inspiral_cut_frequency",
    "printed_inspiral_cut_frequency",
    "compatibility_inspiral_cut_frequency",
    "printed_veto_nodes",
    "normalized_inspiral_amplitude",
    "physical_inspiral_amplitude",
]


SUPPORTED_MODES = ((2, 1), (3, 2), (3, 3), (4, 4))
COLLOCATION_FRACTIONS = (0.5, 0.75, 1.0)
_PSEUDO_PN_POWERS = (7.0 / 3.0, 8.0 / 3.0, 3.0)
PRINTED_CUT_VERSION = "printed-paper"
COMPATIBILITY_CUT_VERSION = "release-compatibility"


def _real_tensors(*values: object) -> tuple[torch.Tensor, ...]:
    tensors = []
    for value in values:
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise TypeError("assembly inputs must be real and not boolean")
        # Recreate non-tensor inputs directly at float64 so Python literals do
        # not first round through torch's default float32 dtype.  Tensor inputs
        # retain their existing precision before any widening conversion.
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if not bool(torch.isfinite(tensor).all().detach().cpu()):
            raise ValueError("assembly inputs must be finite")
        tensors.append(tensor)
    return torch.broadcast_tensors(*tensors)


def _validate_mode(mode: tuple[int, int]) -> None:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"supported clean assembly modes are {SUPPORTED_MODES}")


def _mass_parameters(
    frequency: object,
    f_ins: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
) -> tuple[torch.Tensor, ...]:
    f, cut, mt, mass1, mass2, spin1, spin2 = _real_tensors(
        frequency, f_ins, total_mass_seconds, m1, m2, chi1, chi2
    )
    if bool((f <= 0).any()) or bool((cut <= 0).any()):
        raise ValueError("frequency and f_ins must be positive")
    if bool((mt <= 0).any()) or bool((mass1 <= 0).any()) or bool((mass2 <= 0).any()):
        raise ValueError("total_mass_seconds and component masses must be positive")
    if bool((f > cut).any()):
        raise ValueError("assembly frequencies must satisfy frequency <= f_ins")
    if bool((spin1.abs() > 1).any()) or bool((spin2.abs() > 1).any()):
        raise ValueError("component spins must lie in [-1, 1]")
    q = torch.maximum(mass1, mass2) / torch.minimum(mass1, mass2)
    if bool((q >= 20).any()):
        raise ValueError("the bounded clean inspiral assembly requires q < 20")
    return f, cut, mt, mass1, mass2, spin1, spin2, q


def _cut_version(version: str) -> str:
    if not isinstance(version, str):
        raise TypeError("cut version must be a string")
    normalized = version.strip().lower().replace("_", "-")
    if normalized in {"printed", "paper", "printed-paper", "printedpaper"}:
        return PRINTED_CUT_VERSION
    if normalized in {
        "compatibility",
        "compat",
        "release-compatibility",
        "releasecompatibility",
    }:
        return COMPATIBILITY_CUT_VERSION
    raise ValueError(
        "cut version must be 'printed-paper' or 'release-compatibility'"
    )


def inspiral_cut_frequency(
    mode: tuple[int, int],
    f_meco_22: object,
    f_isco_22: object,
    chi_eff: object,
    *,
    f_rd_32: object | None = None,
    f_rd_22: object | None = None,
    version: str = PRINTED_CUT_VERSION,
) -> torch.Tensor:
    """Return a named XHM ``fcutIns`` convention.

    ``f_meco_22`` and ``f_isco_22`` are positive dimensionless or physical
    22-mode frequencies in one common unit.  The table's 33 entry is copied
    literally: its final term is ``-5/6*chi_eff`` rather than a square.  The
    32 entry uses ``abs(chi_eff)`` and requires explicit ``f_rd_32`` and
    ``f_rd_22`` inputs.  In the printed-paper convention epsilon multiplies
    only the MECO term.  The release-compatibility convention is an explicit
    diagnostic control in which epsilon multiplies the complete parenthesis.
    No MECO/ISCO or ringdown fit is selected here.
    """
    _validate_mode(mode)
    selected_version = _cut_version(version)
    meco, isco, spin = _real_tensors(f_meco_22, f_isco_22, chi_eff)
    if bool((meco <= 0).any()) or bool((isco <= 0).any()):
        raise ValueError("f_meco_22 and f_isco_22 must be positive")
    epsilon_ins = torch.ones_like(meco)
    if mode == (2, 1):
        delta_ins = 3.0 / 4.0 - 0.235 * spin - (5.0 / 6.0) * spin.square()
    elif mode == (3, 3):
        delta_ins = 3.0 / 4.0 - 0.235 * spin - (5.0 / 6.0) * spin
    elif mode == (3, 2):
        if f_rd_32 is None or f_rd_22 is None:
            raise ValueError("(3,2) cut requires f_rd_32 and f_rd_22")
        rd32, rd22 = _real_tensors(f_rd_32, f_rd_22)
        if bool((rd32 <= 0).any()) or bool((rd22 <= 0).any()):
            raise ValueError("f_rd_32 and f_rd_22 must be positive")
        epsilon_ins = rd32 / rd22
        if not bool(torch.isfinite(epsilon_ins).all()):
            raise ValueError("f_rd_32/f_rd_22 must be finite")
        delta_ins = 3.0 / 4.0 - 0.235 * spin.abs()
    else:
        delta_ins = 3.0 / 4.0 - 0.235 * spin
    meco, isco, spin, epsilon_ins = torch.broadcast_tensors(
        meco, isco, spin, epsilon_ins
    )
    if mode == (3, 2) and selected_version == COMPATIBILITY_CUT_VERSION:
        # This is the release-compatibility control from the sanitized audit:
        # the f_RD32/f_RD22 epsilon scales the full MECO-to-ISCO expression.
        cut = (mode[1] / 2.0) * epsilon_ins * (
            meco + (isco - meco).abs() * delta_ins
        )
    else:
        # Keep the paper-literal printed control intact.  For the other modes
        # the named compatibility selector currently has the same cut rule.
        cut = (mode[1] / 2.0) * (
            meco * epsilon_ins + (isco - meco).abs() * delta_ins
        )
    if bool((cut <= 0).any()):
        raise ValueError("inspiral f_ins is non-positive for the supplied inputs")
    return cut


def printed_inspiral_cut_frequency(
    mode: tuple[int, int],
    f_meco_22: object,
    f_isco_22: object,
    chi_eff: object,
    *,
    f_rd_32: object | None = None,
    f_rd_22: object | None = None,
) -> torch.Tensor:
    """Return the paper-literal printed XHM inspiral amplitude cut.

    This is retained as a separate control because the 32 printed expression
    applies ``epsilon=f_rd_32/f_rd_22`` to the MECO term only.  Use
    :func:`compatibility_inspiral_cut_frequency` for the explicit
    release-compatibility control.
    """
    return inspiral_cut_frequency(
        mode,
        f_meco_22,
        f_isco_22,
        chi_eff,
        f_rd_32=f_rd_32,
        f_rd_22=f_rd_22,
        version=PRINTED_CUT_VERSION,
    )


def compatibility_inspiral_cut_frequency(
    mode: tuple[int, int],
    f_meco_22: object,
    f_isco_22: object,
    chi_eff: object,
    *,
    f_rd_32: object | None = None,
    f_rd_22: object | None = None,
) -> torch.Tensor:
    """Return the named release-compatibility inspiral cut control.

    For mode ``(3,2)`` this uses
    ``(m/2)*(f_rd_32/f_rd_22)*(f_meco + |f_isco-f_meco|*delta_ins)``.
    Other modes currently share the printed cut rule.  The helper does not
    choose any ringdown or remnant fit; callers must supply the 32 ringdown
    frequencies explicitly.
    """
    return inspiral_cut_frequency(
        mode,
        f_meco_22,
        f_isco_22,
        chi_eff,
        f_rd_32=f_rd_32,
        f_rd_22=f_rd_22,
        version=COMPATIBILITY_CUT_VERSION,
    )


def _as_three_values(values: object, name: str) -> torch.Tensor:
    raw = torch.as_tensor(values)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(f"{name} must be finite real values")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 0 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have exactly three trailing collocation values")
    if not bool(torch.isfinite(tensor).all().detach().cpu()):
        raise ValueError(f"{name} must be finite real values")
    return tensor


def printed_veto_nodes(
    mode: tuple[int, int],
    q: object,
    chi1: object,
    chi2: object,
    collocation_values: object,
    *,
    geometric_veto_amplitudes: object | None = None,
) -> tuple[int, ...]:
    """Return retained collocation-node indices after the printed vetoes.

    The result is a single tuple for one intrinsic point.  The collocation
    values are normalized ``H_lm`` values.  The 21 vetoes use the actual
    Fourier amplitudes in the paper's geometric-unit convention: callers
    must supply the three ``geometric_veto_amplitudes`` whenever the
    automatic 21 veto is selected.  The normalized values are never silently
    treated as geometric amplitudes.

    The paper's 21 rule (only for ``q<8``) removes values below 0.2; it also
    removes the middle node whenever the three actual amplitudes are not
    monotonic.  Its 32 q<20 parameter regions remove the two highest nodes
    for ``q>2.5, chi1<-0.6, chi2>0`` or for both spins below ``-0.9``; the
    dynamic ringdown-amplitude threshold is outside this helper's inputs.
    Its 33 special region removes the last node for ``1<q<1.2``,
    ``chi1<-0.1``, and ``chi2>0``.  No 44 veto is printed in the cited
    section.  Removing nodes is implemented by the caller as removal of the
    highest pseudo-PN powers.
    """
    _validate_mode(mode)
    q_t, spin1, spin2 = _real_tensors(q, chi1, chi2)
    values = _as_three_values(collocation_values, "collocation_values")
    if q_t.numel() != 1 or spin1.numel() != 1 or spin2.numel() != 1:
        raise ValueError("printed_veto_nodes accepts one intrinsic point")
    if values.numel() != 3:
        raise ValueError("printed_veto_nodes accepts one set of three values")
    q_value = float(q_t.item())
    if q_value < 1:
        raise ValueError("q must be at least one")
    values = values.reshape(3)
    veto_values = None if geometric_veto_amplitudes is None else _as_three_values(
        geometric_veto_amplitudes, "geometric_veto_amplitudes"
    ).reshape(3)
    removed: set[int] = set()
    if mode == (2, 1):
        if veto_values is None:
            raise ValueError(
                "explicit geometric_veto_amplitudes are required for automatic 21 vetoes"
            )
        if q_value < 8:
            removed.update(int(i) for i in torch.where(veto_values < 0.2)[0].tolist())
        differences = veto_values[1:] - veto_values[:-1]
        monotonic = bool((differences >= 0).all()) or bool((differences <= 0).all())
        if not monotonic:
            removed.add(1)
    if mode == (3, 2):
        # These are the q<20 portions of the published 32 parameter vetoes.
        # The separate ringdown-amplitude threshold is not inferable from the
        # three inspiral values accepted by this clean helper.
        if q_value > 2.5 and float(spin1.item()) < -0.6 and float(spin2.item()) > 0:
            removed.update((1, 2))
        if float(spin1.item()) < -0.9 and float(spin2.item()) < -0.9:
            removed.update((1, 2))
        differences = values[1:] - values[:-1]
        monotonic = bool((differences >= 0).all()) or bool((differences <= 0).all())
        if not monotonic:
            removed.add(1)
    if mode == (3, 3) and 1 < q_value < 1.2:
        if float(spin1.item()) < -0.1 and float(spin2.item()) > 0:
            removed.add(2)
    return tuple(index for index in range(3) if index not in removed)


def _pn_normalized(
    mode: tuple[int, int],
    frequency: torch.Tensor,
    total_mass_seconds: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
) -> torch.Tensor:
    if mode == (2, 1):
        # ``normalized_spa_amplitude_21`` is |Hhat_21|, whereas Appendix E's
        # common H_lm uses |Hhat_lm|*(m/2)**(7/6).  For m=1 this factor is
        # 2**(-7/6) before the A22^0-normalized ansatz is assembled.
        return 2.0 ** (-7.0 / 6.0) * normalized_spa_amplitude_21(
            frequency, total_mass_seconds, m1, m2, chi1, chi2
        )
    if mode in ((3, 2), (4, 4)):
        return normalized_pn_amplitude_compatibility(
            mode, frequency, total_mass_seconds, m1, m2, chi1, chi2
        )
    return normalized_pn_amplitude(
        mode, frequency, total_mass_seconds, m1, m2, chi1, chi2
    )


def normalized_inspiral_amplitude(
    mode: tuple[int, int],
    frequency: object,
    f_ins: object,
    collocation_values: object,
    total_mass_seconds: object,
    m1: object,
    m2: object,
    chi1: object,
    chi2: object,
    *,
    retained_nodes: Sequence[int] | None = None,
    geometric_veto_amplitudes: object | None = None,
    apply_vetoes: bool = True,
) -> torch.Tensor:
    """Evaluate the clean, dimensionless XHM inspiral amplitude ansatz.

    ``collocation_values`` are explicit values of ``mathcal H_lm`` at
    ``(.5,.75,1)*f_ins`` and are therefore dimensionless.  The PN baseline is
    the reviewed clean Appendix-E expression: non-reexpanded SPA for 21 and
    re-expanded Appendix-E amplitudes for 32/33/44, with the named
    compatibility signs for 32/44.  The returned magnitude is
    dimensionless and keeps no distance or total-mass scale.  Use
    ``physical_inspiral_amplitude`` for the Fourier-domain ``M**2/D`` scale.

    ``retained_nodes`` bypasses automatic veto selection and explicitly names
    the retained rows.  The collocation solver then keeps the same number of
    lowest pseudo-PN powers and zero-pads the rest.  With no explicit tuple,
    the printed mode/q/spin vetoes are applied when ``apply_vetoes`` is true.
    Automatic veto selection is intentionally scalar; batched evaluations use
    a shared explicit ``retained_nodes`` tuple.
    """
    _validate_mode(mode)
    values = _as_three_values(collocation_values, "collocation_values")
    f, cut, mt, mass1, mass2, spin1, spin2, q = _mass_parameters(
        frequency, f_ins, total_mass_seconds, m1, m2, chi1, chi2
    )
    if retained_nodes is None:
        if apply_vetoes:
            # A frequency grid may be batched while the intrinsic point is
            # scalar.  Select the paper veto from the unbroadcast intrinsic
            # inputs so this useful scalar-intrinsic/vector-frequency case is
            # distinct from a genuinely batched intrinsic evaluation.
            intrinsic = _real_tensors(
                total_mass_seconds, m1, m2, chi1, chi2
            )
            _, mass1_i, mass2_i, spin1_i, spin2_i = intrinsic
            if any(item.numel() != 1 for item in intrinsic):
                raise ValueError(
                    "automatic veto selection requires scalar intrinsic parameters; "
                    "pass retained_nodes for a batch"
                )
            q_i = torch.maximum(mass1_i, mass2_i) / torch.minimum(mass1_i, mass2_i)
            # The printed 33 veto uses chi1/chi2 in the convention where
            # body 1 is the heavier object.  Canonicalize before applying
            # that asymmetric spin-region test so exchanged inputs agree.
            if float(mass1_i.item()) >= float(mass2_i.item()):
                veto_spin1, veto_spin2 = spin1_i, spin2_i
            else:
                veto_spin1, veto_spin2 = spin2_i, spin1_i
            retained = printed_veto_nodes(
                mode,
                q_i,
                veto_spin1,
                veto_spin2,
                values,
                geometric_veto_amplitudes=geometric_veto_amplitudes,
            )
        else:
            retained = (0, 1, 2)
    else:
        retained = tuple(retained_nodes)
        if len(set(retained)) != len(retained) or any(node not in (0, 1, 2) for node in retained):
            raise ValueError("retained_nodes must contain distinct indices from {0,1,2}")

    node_fraction = torch.as_tensor(
        COLLOCATION_FRACTIONS, dtype=torch.float64, device=cut.device
    )
    node_frequency = cut.unsqueeze(-1) * node_fraction
    node_pn = _pn_normalized(
        mode,
        node_frequency,
        mt.unsqueeze(-1),
        mass1.unsqueeze(-1),
        mass2.unsqueeze(-1),
        spin1.unsqueeze(-1),
        spin2.unsqueeze(-1),
    )
    node_pn, values = torch.broadcast_tensors(node_pn, values)
    coefficients = collocation_coefficients(
        node_pn,
        values,
        cut,
        active_terms=len(retained),
        retained_nodes=retained,
    )
    u = f / cut
    basis = torch.stack(tuple(u.pow(power) for power in _PSEUDO_PN_POWERS), dim=-1)
    pn = _pn_normalized(mode, f, mt, mass1, mass2, spin1, spin2)
    return pn + (coefficients * basis).sum(dim=-1)


def physical_inspiral_amplitude(
    mode: tuple[int, int],
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
    retained_nodes: Sequence[int] | None = None,
    geometric_veto_amplitudes: object | None = None,
    apply_vetoes: bool = True,
) -> torch.Tensor:
    """Return the raw physical Fourier-domain amplitude magnitude.

    Collocation values remain dimensionless ``mathcal H_lm`` values.  The
    result multiplies them by
    ``(M**2/D)*pi*sqrt(2*eta/3)*(pi*f*M)**(-7/6)`` from the paper's
    ``A22^0`` normalization, so absolute ``M**2/D`` scale is retained.  All
    frequencies and masses are SI-consistent through ``total_mass_seconds``
    and ``distance_seconds``.
    """
    normalized = normalized_inspiral_amplitude(
        mode,
        frequency,
        f_ins,
        collocation_values,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        retained_nodes=retained_nodes,
        geometric_veto_amplitudes=geometric_veto_amplitudes,
        apply_vetoes=apply_vetoes,
    )
    f, _, mt, mass1, mass2, _, _ = _real_tensors(
        frequency, f_ins, total_mass_seconds, m1, m2, chi1, chi2
    )
    distance, = _real_tensors(distance_seconds)
    f, mt, mass1, mass2, distance = torch.broadcast_tensors(
        f, mt, mass1, mass2, distance
    )
    if bool((distance <= 0).any()):
        raise ValueError("distance_seconds must be positive")
    eta = mass1 * mass2 / (mass1 + mass2).square()
    a22_0 = (
        mt.square()
        / distance
        * math.pi
        * torch.sqrt(2.0 * eta / 3.0)
        * (math.pi * f * mt).pow(-7.0 / 6.0)
    )
    return a22_0 * normalized
