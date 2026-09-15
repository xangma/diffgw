"""Clean-room diagnosis of the near-degenerate MSA cubic stage.

This is a bounded, read-only arithmetic diagnostic.  It compares the current
published Appendix-B coefficient evaluation with one algebraically equivalent
grouped-``Delta`` evaluation, and keeps two small root-arithmetic controls
separate from that candidate: an ``atan2`` evaluation of the same
trigonometric cubic angle and a companion-matrix solve.  The latter controls
are diagnostic only; no solver, threshold, or physical convention is chosen
from them.

The comparison uses the local clean MSA helpers and public
``SimIMRPhenomXPMSAAngles``/``SimInspiralChooseFDWaveform`` interfaces.  It
does not fit a residual, use a reference implementation, or edit runtime
code.  The output report is the only file written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import sympy as sp
import torch

import torchwave.approximants.precession_msa_angles_clean as angle_module
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles,
)
from torchwave.approximants.precession_msa_initial_clean import (
    cubic_coefficients as direct_cubic_coefficients,
    initial_invariants,
)
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223


# Save the unwrapped clean masks before the in-process diagnostic monkeypatches
# below.  This is still the existing helper, not a second implementation.
_clean_safe_cubic_roots = angle_module._safe_cubic_roots


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "msa-root-boundary-clean"
)
DEFAULT_OUTPUT = REPORT_DIR / "report.json"
PRIOR_PRECISION_REPORT = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "msa-boundary-contract-clean" / "msa_boundary_precision_report.json"
)

MTSUN_SI = 4.925490947641267e-6
F_REF = 20.0
ORDER = 5
MPRIME = 2
CONVENTION = 1
FINAL_SPIN_MOD = 3
DISTANCE = 500.0
INCLINATION = 0.7
PHIC = 0.3
TC = 0.0
ROOT_FREQUENCIES = torch.tensor(
    [185.0, 190.0, 196.0, 200.0], dtype=torch.float64
)
WAVEFORM_FREQUENCIES = torch.arange(20.0, 200.25, 1.0, dtype=torch.float64)

# The first fixture is the prior clean-room boundary row.  The other two are
# independently selected controls: one remains close to the root gap while
# changing the spins, and one has ordinary, non-degenerate generic geometry.
CASES = (
    {
        "id": "audited_tiny_in_plane_35_25",
        "role": "audited known boundary failure",
        "mass1": 35.0,
        "mass2": 25.0,
        "spin1": (1.0e-5, 0.0, 0.2),
        "spin2": (0.0, 1.0e-5, -0.1),
    },
    {
        "id": "fresh_nearby_small_in_plane_37_23",
        "role": "fresh nearby root-gap control",
        "mass1": 37.0,
        "mass2": 23.0,
        "spin1": (2.0e-5, -1.0e-5, 0.27),
        "spin2": (-1.2e-5, 1.7e-5, -0.13),
    },
    {
        "id": "fresh_generic_42_18",
        "role": "fresh generic non-degenerate control",
        "mass1": 42.0,
        "mass2": 18.0,
        "spin1": (0.23, -0.17, 0.31),
        "spin2": (-0.11, 0.19, -0.22),
    },
)


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


RUNTIME_FILES = (
    "src/torchwave/approximants/precession_msa_initial_clean.py",
    "src/torchwave/approximants/precession_msa_angles_clean.py",
    "src/torchwave/approximants/precession_msa_oscillatory_clean.py",
    "src/torchwave/approximants/precession_msa_waveform_clean.py",
)


def _runtime_hashes():
    return {name: _sha256(ROOT / name) for name in RUNTIME_FILES}


def _json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _float_tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _flatten_floats(value):
    return [float(x) for x in torch.as_tensor(value).detach().cpu().reshape(-1)]


def _max_abs(value):
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().abs().max())


def _relative_l2(value, target):
    value = np.asarray(value)
    target = np.asarray(target)
    denom = np.linalg.norm(target)
    return float(np.linalg.norm(value - target) / denom) if denom else 0.0


def _complex_metrics(actual, target):
    actual = np.asarray(actual, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    delta = actual - target
    denom = np.linalg.norm(target)
    return {
        "difference_l2": float(np.linalg.norm(delta)),
        "target_l2": float(denom),
        "relative_l2": float(np.linalg.norm(delta) / denom) if denom else 0.0,
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def _options(convention=CONVENTION):
    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 223)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(
        options, ORDER
    )
    lalsim.SimInspiralWaveformParamsInsertPhenomXPConvention(
        options, int(convention)
    )
    lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(
        options, FINAL_SPIN_MOD
    )
    return options


def _public_angles(case, frequencies, convention=CONVENTION):
    vector = lal.CreateREAL8Vector(len(frequencies))
    vector.data[:] = [float(x) for x in frequencies]
    result = lalsim.SimIMRPhenomXPMSAAngles(
        vector,
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        *case["spin1"], *case["spin2"],
        INCLINATION, F_REF, MPRIME, _options(convention),
    )
    return tuple(np.asarray(value.data, dtype=np.float64) for value in result)


def _public_waveform(case, convention=CONVENTION):
    flow = float(WAVEFORM_FREQUENCIES[0])
    fhigh = float(WAVEFORM_FREQUENCIES[-1])
    df = float(WAVEFORM_FREQUENCIES[1] - WAVEFORM_FREQUENCIES[0])
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        case["mass1"] * lal.MSUN_SI,
        case["mass2"] * lal.MSUN_SI,
        *case["spin1"], *case["spin2"],
        DISTANCE * 1.0e6 * lal.PC_SI,
        INCLINATION, PHIC, 0.0, 0.0, 0.0, df, flow, fhigh, F_REF,
        _options(convention), lalsim.IMRPhenomXP,
    )
    start = int(round(flow / df))
    stop = start + WAVEFORM_FREQUENCIES.numel()
    return (
        np.asarray(hp.data.data[start:stop], dtype=np.complex128),
        np.asarray(hc.data.data[start:stop], dtype=np.complex128),
    )


def _grouped_delta_coefficients(q, eta, delta_m, s1sq, s2sq, L, J, xi):
    """Appendix-B coefficients after the exact substitution Delta=J²-L²."""
    q, eta, dm, s1, s2, L, J, xi = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=torch.float64)
          for x in (q, eta, delta_m, s1sq, s2sq, L, J, xi))
    )
    delta = J.square() - L.square()
    B = (
        L.square() * (q + q.reciprocal()) - 2.0 * J.square()
        + 2.0 * L * xi + (q - 1.0) * s1
        + (q.reciprocal() - 1.0) * s2
    )
    C = (
        delta.square() - 2.0 * L * xi * delta
        + 4.0 * eta * L.square() * xi.square()
        - 2.0 * dm * (s1 - s2) * xi * L
        + 2.0 * (1.0 - q) / q * (q * s1 - s2) * delta
        - 2.0 * (1.0 - q).square() / q * (s1 + s2) * L.square()
    )
    D = (
        (1.0 - q) / q * (s2 - q * s1) * delta.square()
        + dm.square() / eta * (s1 - s2).square() * L.square()
        + 2.0 * dm * L * xi * (s1 - s2) * delta
    )
    return B, C, D


def _atan2_cubic_roots(B, C, D):
    """Current clean root masks with atan2(theta) in place of acos(theta)."""
    B, C, D = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=torch.float64) for x in (B, C, D))
    )
    finite = torch.isfinite(B) & torch.isfinite(C) & torch.isfinite(D)
    B_eval = torch.where(finite, B, torch.zeros_like(B))
    C_eval = torch.where(finite, C, torch.zeros_like(C))
    D_eval = torch.where(finite, D, torch.zeros_like(D))
    p = C - B.square() / 3.0
    p_eval = C_eval - B_eval.square() / 3.0
    p_safe = torch.where(
        torch.isfinite(p_eval) & (p_eval < 0.0), p_eval,
        -torch.ones_like(p_eval),
    )
    q_eval = 2.0 * B_eval.pow(3) / 27.0 - B_eval * C_eval / 3.0 + D_eval
    raw = 3.0 * q_eval / (2.0 * p_safe) * torch.sqrt(-3.0 / p_safe)
    raw_finite = torch.isfinite(raw)
    raw_safe = torch.where(raw_finite, raw, torch.zeros_like(raw))
    raw_clamped = raw_safe.clamp(-1.0, 1.0)
    endpoint = raw_clamped.abs() >= 1.0
    raw_for_atan2 = torch.where(endpoint, raw_clamped.detach(), raw_clamped)
    sine = torch.sqrt(torch.clamp(1.0 - raw_for_atan2.square(), min=0.0))
    theta = torch.atan2(sine, raw_for_atan2) / 3.0
    k = torch.arange(3, dtype=B.dtype, device=B.device)
    roots = -B[..., None] / 3.0 + 2.0 * torch.sqrt(
        -p_safe[..., None] / 3.0
    ) * torch.cos(theta[..., None] - 2.0 * torch.pi * k / 3.0)
    roots = torch.sort(roots, dim=-1).values
    physical = torch.sort(roots[..., 1:].abs(), dim=-1).values
    roots = torch.cat((roots[..., :1], physical), dim=-1)
    valid = finite & torch.isfinite(p) & (p < 0.0) & raw_finite
    valid = valid & (raw.abs() <= 1.0 + 1.0e-10)
    gap = (roots[..., 2] - roots[..., 1]).abs()
    degenerate = valid & torch.isfinite(gap) & (gap < 1.0e-5)
    return roots, valid, degenerate


def _companion_roots(B, C, D):
    """Companion-matrix control with masks from the clean trigonometric path."""
    baseline_roots, valid, degenerate = _clean_safe_cubic_roots(B, C, D)
    B, C, D = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=torch.float64) for x in (B, C, D))
    )
    matrix = torch.zeros((*B.shape, 3, 3), dtype=B.dtype, device=B.device)
    matrix[..., 0, 2] = -D
    matrix[..., 1, 0] = 1.0
    matrix[..., 1, 2] = -C
    matrix[..., 2, 1] = 1.0
    matrix[..., 2, 2] = -B
    eig = torch.linalg.eigvals(matrix)
    roots = torch.sort(eig.real, dim=-1).values
    physical = torch.sort(roots[..., 1:].abs(), dim=-1).values
    roots = torch.cat((roots[..., :1], physical), dim=-1)
    # Keep the clean validity/degeneracy convention so this remains an
    # arithmetic control, rather than introducing a new physical branch.
    del baseline_roots
    return roots, valid, degenerate


def _symbolic_identity():
    q, eta, dm, s1, s2, L, J, xi = sp.symbols(
        "q eta dm s1 s2 L J xi", nonzero=True
    )
    delta = J**2 - L**2
    direct_b = (L**2 + s1) * q + 2 * L * xi - 2 * J**2 - s1 - s2 + (L**2 + s2) / q
    grouped_b = L**2 * (q + 1 / q) - 2 * J**2 + 2 * L * xi + (q - 1) * s1 + (1 / q - 1) * s2
    direct_c = (
        delta**2 - 2 * L * xi * delta
        - 2 * (1 - q) / q * (s1 - q * s2) * L**2
        + 4 * eta * L**2 * xi**2 - 2 * dm * (s1 - s2) * xi * L
        + 2 * (1 - q) / q * (q * s1 - s2) * J**2
    )
    grouped_c = (
        delta**2 - 2 * L * xi * delta + 4 * eta * L**2 * xi**2
        - 2 * dm * (s1 - s2) * xi * L
        + 2 * (1 - q) / q * (q * s1 - s2) * delta
        - 2 * (1 - q)**2 / q * (s1 + s2) * L**2
    )
    direct_d = (
        (1 - q) / q * (s2 - q * s1) * delta**2
        + dm**2 / eta * (s1 - s2)**2 * L**2
        + 2 * dm * L * xi * (s1 - s2) * delta
    )
    grouped_d = direct_d
    return {
        "B_difference_simplified": str(sp.simplify(direct_b - grouped_b)),
        "C_difference_simplified": str(sp.simplify(direct_c - grouped_c)),
        "D_difference_simplified": str(sp.simplify(direct_d - grouped_d)),
        "identity": "All three differences simplify exactly to zero.",
    }


def _numeric_identity_check():
    rng = np.random.default_rng(223)
    rows = []
    for _ in range(8):
        q = float(rng.uniform(0.15, 0.95))
        eta = q / (1.0 + q) ** 2
        dm = (1.0 - q) / (1.0 + q)
        s1, s2 = (float(rng.uniform(0.0, 0.2)) for _ in range(2))
        L = float(rng.uniform(0.2, 0.8))
        J = float(rng.uniform(0.2, 0.9))
        xi = float(rng.uniform(-0.1, 0.1))
        current = direct_cubic_coefficients(q, eta, dm, s1, s2, L, J, xi)
        grouped = _grouped_delta_coefficients(q, eta, dm, s1, s2, L, J, xi)
        rows.append({
            "max_abs_float64_difference": _max_abs(
                torch.stack([a - b for a, b in zip(current, grouped)])
            ),
            "inputs": [q, eta, dm, s1, s2, L, J, xi],
        })
    return {
        "sample_count": len(rows),
        "max_abs_float64_difference": max(
            row["max_abs_float64_difference"] for row in rows
        ),
        "samples": rows,
        "interpretation": (
            "The symbolic identity is exact; the nonzero sample differences "
            "are solely operation-order roundoff from evaluating equivalent "
            "expressions in binary64."
        ),
    }


def _state(case):
    m1 = torch.tensor(case["mass1"], dtype=torch.float64)
    m2 = torch.tensor(case["mass2"], dtype=torch.float64)
    s1 = torch.tensor(case["spin1"], dtype=torch.float64)
    s2 = torch.tensor(case["spin2"], dtype=torch.float64)
    v0 = torch.tensor(
        math.pi * (case["mass1"] + case["mass2"]) * MTSUN_SI * F_REF,
        dtype=torch.float64,
    ).pow(1.0 / 3.0)
    L0 = (m1 * m2 / (m1 + m2).square()) / v0
    orbital = torch.zeros(3, dtype=torch.float64)
    orbital[2] = L0
    q, eta, delta_m, S1, S2, L, J, xi = initial_invariants(
        m1, m2, s1, s2, orbital
    )
    B, C, D = direct_cubic_coefficients(
        q, eta, delta_m, S1.square(), S2.square(), L, J, xi
    )
    return {
        "q": float(q),
        "eta": float(eta),
        "delta_m": float(delta_m),
        "S1": float(S1),
        "S2": float(S2),
        "L0": float(L),
        "J0": float(J),
        "xi": float(xi),
        "initial_cubic_direct": [float(B), float(C), float(D)],
    }


def _call_record(B, C, D, roots, valid, degenerate):
    B, C, D = (torch.as_tensor(x).detach().cpu() for x in (B, C, D))
    roots = torch.as_tensor(roots).detach().cpu()
    residual = roots.pow(3) + B[..., None] * roots.pow(2) + C[..., None] * roots + D[..., None]
    return {
        "B": B.tolist(),
        "C": C.tolist(),
        "D": D.tolist(),
        "roots": roots.tolist(),
        "valid": torch.as_tensor(valid).detach().cpu().tolist(),
        "degenerate": torch.as_tensor(degenerate).detach().cpu().tolist(),
        "max_abs_polynomial_residual": float(residual.abs().max()),
    }


def _run_angles(case, variant_name, coefficient_fn, root_fn):
    calls = []
    original_coeff = angle_module.cubic_coefficients
    original_roots = angle_module._safe_cubic_roots

    def wrapped_roots(B, C, D):
        roots, valid, degenerate = root_fn(B, C, D)
        calls.append(_call_record(B, C, D, roots, valid, degenerate))
        return roots, valid, degenerate

    angle_module.cubic_coefficients = coefficient_fn
    angle_module._safe_cubic_roots = wrapped_roots
    try:
        with torch.no_grad():
            output = assemble_precession_msa_angles(
                ROOT_FREQUENCIES,
                case["mass1"], case["mass2"],
                torch.tensor(case["spin1"], dtype=torch.float64),
                torch.tensor(case["spin2"], dtype=torch.float64),
                f_ref=F_REF, order=ORDER, compatibility="v223",
            )
    finally:
        angle_module.cubic_coefficients = original_coeff
        angle_module._safe_cubic_roots = original_roots

    result = {
        "variant": variant_name,
        "calls": calls,
        "roots_reference": output["roots_reference"].tolist(),
        "roots": output["roots"].tolist(),
        "c1": float(output["c1"]),
        "sav": float(output["sav"]),
        "frequency_degenerate": output["frequency_degenerate"].tolist(),
        "rational_lambda_a": output["rational_lambda_a"].tolist(),
        "rational_lambda_b": output["rational_lambda_b"].tolist(),
        "alpha": output["alpha"].tolist(),
        "gamma": output["gamma"].tolist(),
        "cos_beta": output["cos_beta"].tolist(),
    }
    return result


def _run_waveform(case, variant_name, coefficient_fn=None, root_fn=None):
    original_coeff = angle_module.cubic_coefficients
    original_roots = angle_module._safe_cubic_roots
    if coefficient_fn is not None:
        angle_module.cubic_coefficients = coefficient_fn
    if root_fn is not None:
        angle_module._safe_cubic_roots = root_fn
    try:
        with torch.no_grad():
            hp, hc = waveform_v223(
                WAVEFORM_FREQUENCIES,
                case["mass1"], case["mass2"],
                *case["spin1"], *case["spin2"],
                distance=DISTANCE, inclination=INCLINATION,
                phic=PHIC, tc=TC, f_ref=F_REF,
                convention=CONVENTION, order=ORDER,
            )
    finally:
        angle_module.cubic_coefficients = original_coeff
        angle_module._safe_cubic_roots = original_roots
    return {
        "variant": variant_name,
        "hp": np.asarray(hp.detach().cpu(), dtype=np.complex128),
        "hc": np.asarray(hc.detach().cpu(), dtype=np.complex128),
    }


def _angle_public_metrics(case, native):
    public = _public_angles(case, ROOT_FREQUENCIES.tolist(), CONVENTION)
    native_alpha = np.asarray(native["alpha"], dtype=np.float64)
    native_gamma = np.asarray(native["gamma"], dtype=np.float64)
    native_cos = np.asarray(native["cos_beta"], dtype=np.float64)
    alpha = public[0] - native_alpha
    alpha_offset = float(alpha[0])
    return {
        "public_convention": CONVENTION,
        "alpha_gauge_offset": alpha_offset,
        "alpha_raw_max_abs": float(np.max(np.abs(alpha))),
        "alpha_gauge_aligned_max_abs": float(np.max(np.abs(alpha - alpha_offset))),
        "gamma_raw_max_abs": float(np.max(np.abs(native_gamma - public[1]))),
        "cos_beta_raw_max_abs": float(np.max(np.abs(native_cos - public[2]))),
        "public_alpha": public[0].tolist(),
        "public_gamma": public[1].tolist(),
        "public_cos_beta": public[2].tolist(),
    }


def _variant_delta(baseline, candidate):
    result = {
        "initial_roots_minus_direct": [],
        "current_roots_minus_direct": [],
        "lambda_a_minus_direct": [],
        "lambda_b_minus_direct": [],
        "alpha_minus_direct": [],
        "gamma_minus_direct": [],
        "cos_beta_minus_direct": [],
    }
    for key, out_key in (
        ("roots_reference", "initial_roots_minus_direct"),
        ("roots", "current_roots_minus_direct"),
        ("rational_lambda_a", "lambda_a_minus_direct"),
        ("rational_lambda_b", "lambda_b_minus_direct"),
        ("alpha", "alpha_minus_direct"),
        ("gamma", "gamma_minus_direct"),
        ("cos_beta", "cos_beta_minus_direct"),
    ):
        a = np.asarray(candidate[key], dtype=np.float64)
        b = np.asarray(baseline[key], dtype=np.float64)
        result[out_key] = (a - b).tolist()
        result[out_key.replace("_minus_direct", "_max_abs")] = float(
            np.max(np.abs(a - b), initial=0.0)
        )
    return result


def _serialize_waveform(result):
    return {
        "variant": result["variant"],
        "hp_real": result["hp"].real.tolist(),
        "hp_imag": result["hp"].imag.tolist(),
        "hc_real": result["hc"].real.tolist(),
        "hc_imag": result["hc"].imag.tolist(),
    }


def _prior_precision_evidence():
    if not PRIOR_PRECISION_REPORT.is_file():
        return {"available": False, "path": str(PRIOR_PRECISION_REPORT)}
    prior = json.loads(PRIOR_PRECISION_REPORT.read_text())
    rows = {}
    for key in ("190", "196"):
        row = prior.get("target_rows", {}).get(key)
        if not row:
            continue
        rows[key] = {
            "clean_roots": row.get("clean", {}).get("roots"),
            "independent_mp_roots": row.get("mp", {}).get("current", {}).get("roots"),
            "clean_lambda_a": row.get("clean", {}).get("lambda_a"),
            "independent_mp_lambda_a": row.get("mp_pole_same_mp_roots", {}).get("lambda_a"),
            "clean_minus_mp_root": row.get("differences", {}).get("clean_current_roots_minus_exact_mp"),
        }
    return {
        "available": True,
        "path": str(PRIOR_PRECISION_REPORT.relative_to(ROOT)),
        "source_sha256": _sha256(PRIOR_PRECISION_REPORT),
        "rows": rows,
        "interpretation": (
            "This report reuses the prior fixed-input precision result; it "
            "does not repeat the 50/80/120-digit experiment."
        ),
    }


def _run_case(case):
    variants = (
        ("current_direct", direct_cubic_coefficients, angle_module._safe_cubic_roots),
        ("grouped_delta", _grouped_delta_coefficients, angle_module._safe_cubic_roots),
        ("grouped_delta_atan2", _grouped_delta_coefficients, _atan2_cubic_roots),
        ("companion_control", direct_cubic_coefficients, _companion_roots),
    )
    angle_results = [
        _run_angles(case, name, coeff_fn, root_fn)
        for name, coeff_fn, root_fn in variants
    ]
    baseline = angle_results[0]
    for result in angle_results[1:]:
        result["delta_from_current_direct"] = _variant_delta(baseline, result)

    public_angles = _angle_public_metrics(case, baseline)
    baseline_waveform = _run_waveform(case, "current_direct")
    grouped_waveform = _run_waveform(
        case, "grouped_delta", _grouped_delta_coefficients,
    )
    public_hp, public_hc = _public_waveform(case, CONVENTION)
    waveform_results = {
        "frequencies_hz": WAVEFORM_FREQUENCIES.tolist(),
        "public_target": {
            "hp_hc_metrics_current_direct": {
                "hp": _complex_metrics(baseline_waveform["hp"], public_hp),
                "hc": _complex_metrics(baseline_waveform["hc"], public_hc),
            },
            "hp_hc_metrics_grouped_delta": {
                "hp": _complex_metrics(grouped_waveform["hp"], public_hp),
                "hc": _complex_metrics(grouped_waveform["hc"], public_hc),
            },
        },
        "grouped_minus_current_direct": {
            "hp": _complex_metrics(grouped_waveform["hp"], baseline_waveform["hp"]),
            "hc": _complex_metrics(grouped_waveform["hc"], baseline_waveform["hc"]),
        },
    }
    return {
        "id": case["id"],
        "role": case["role"],
        "parameters": {
            "mass1_msun": case["mass1"],
            "mass2_msun": case["mass2"],
            "spin1": list(case["spin1"]),
            "spin2": list(case["spin2"]),
            "f_ref_hz": F_REF,
            "distance_mpc": DISTANCE,
            "inclination": INCLINATION,
            "phic": PHIC,
            "tc": TC,
            "prec_version": 223,
            "expansion_order": ORDER,
            "convention": CONVENTION,
            "final_spin_mod": FINAL_SPIN_MOD,
        },
        "root_frequencies_hz": ROOT_FREQUENCIES.tolist(),
        "initial_state": _state(case),
        "variants": angle_results,
        "public_angle_metrics_current_direct": public_angles,
        "waveform": waveform_results,
        "known_boundary_interpretation": (
            "The audited tiny case is retained as a visible known failure. "
            "No threshold or row is masked by this diagnostic."
            if "audited" in case["id"] else None
        ),
    }


def main(output=DEFAULT_OUTPUT):
    hashes_before = _runtime_hashes()
    report = {
        "status": "complete_read_only_root_arithmetic_diagnostic",
        "provenance": (
            "Clean-room diagnostic using only local clean MSA modules, the "
            "prior permitted precision report, primary published MSA cubic "
            "equations (1703.03967v3 Eq. 21 / Appendix B), and public LAL "
            "XP angle/waveform callables. No reference implementation, "
            "oracle fitting, threshold change, runtime edit, or long waveform "
            "job is used."
        ),
        "candidate_contract": {
            "runtime_edit_status": "none_selected_pending_root_review",
            "current": (
                "Use the published direct B,C,D expressions and existing "
                "trigonometric root/physical-ordering path."
            ),
            "grouped_delta": (
                "Set Delta=J^2-L^2 once; evaluate B=L^2(q+1/q)-2J^2+2Lxi "
                "+(q-1)S1^2+(1/q-1)S2^2 and C=Delta^2-2LxiDelta "
                "+4eta L^2xi^2-2dm(S1^2-S2^2)xiL "
                "+2(1-q)/q(qS1^2-S2^2)Delta "
                "-2(1-q)^2/q(S1^2+S2^2)L^2; D is the same expression "
                "with Delta."
            ),
            "invariants": (
                "The grouped form is an exact algebraic rewrite only; it "
                "does not alter physical roots, ordering, degeneracy masks, "
                "pole continuation, or endpoint behavior."
            ),
            "controls": (
                "atan2 and companion roots are arithmetic controls, not "
                "production proposals."
            ),
        },
        "equation_identity": _symbolic_identity(),
        "numeric_identity_control": _numeric_identity_check(),
        "prior_precision_evidence": _prior_precision_evidence(),
        "runtime_hashes_before": hashes_before,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sympy": sp.__version__,
            "frequency_root_grid_hz": ROOT_FREQUENCIES.tolist(),
            "waveform_grid_hz": [
                float(WAVEFORM_FREQUENCIES[0]),
                float(WAVEFORM_FREQUENCIES[-1]),
                float(WAVEFORM_FREQUENCIES[1] - WAVEFORM_FREQUENCIES[0]),
            ],
        },
        "cases": [_run_case(case) for case in CASES],
    }
    report["runtime_hashes_after"] = _runtime_hashes()
    report["runtime_unchanged"] = report["runtime_hashes_before"] == report["runtime_hashes_after"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "runtime_unchanged": report["runtime_unchanged"],
        "cases": [case["id"] for case in report["cases"]],
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.output)
