#!/usr/bin/env python3
"""Validate the independent numerical QNM representation used by PhenomD.

The script records numerical checks only.  The rounded ordinates are regenerated
at import time from TorchWave's independent Kerr solver data; no external QNM
table or waveform coefficient is read.  Public LALSimulation calls are used
only for the four existing D gate cases and for a finite-output control sweep.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.kerr_qnm_d_sampling_clean import (
    COMPATIBILITY_SPIN_MAX,
    COMPATIBILITY_SPIN_MIN,
    ENDPOINT_POLICY,
    GENERATED_SPIN_LIMIT,
    KNOT_SPINS,
    N_KNOTS,
    ROUNDED_CYCLIC_ORDINATES,
    ROUNDING_SIG_DIGITS,
    _round_significant,
    _rounding_step,
    evaluate_d_qnm_220,
    independent_cyclic_qnm_220,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "verification" / "d-sampling-clean" / "diagnostic.json"


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _rounding_report() -> Dict[str, Any]:
    """Check every fixed row against its generated value and interval."""

    spins = torch.tensor(KNOT_SPINS, dtype=torch.float64)
    real, damping = independent_cyclic_qnm_220(spins)
    raw = torch.stack((real, damping), dim=-1).tolist()
    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, int, int]] = []
    valid = True
    minimum_margin = math.inf
    for index, (spin, raw_row, rounded_row) in enumerate(
        zip(KNOT_SPINS, raw, ROUNDED_CYCLIC_ORDINATES)
    ):
        channels = []
        for channel, (raw_value, rounded_value) in enumerate(zip(raw_row, rounded_row)):
            expected = _round_significant(float(raw_value), ROUNDING_SIG_DIGITS)
            half_step = _rounding_step(float(raw_value), ROUNDING_SIG_DIGITS)
            error = abs(float(raw_value) - float(rounded_value))
            interval_ok = error <= half_step + 1e-15
            margin = max(0.0, (half_step - error) / half_step)
            minimum_margin = min(minimum_margin, margin)
            candidates.append((margin, index, channel))
            valid = valid and expected == float(rounded_value) and interval_ok
            channels.append(
                {
                    "raw_cyclic": float(raw_value),
                    "rounded_cyclic": float(rounded_value),
                    "expected_rounding": expected,
                    "half_step": half_step,
                    "absolute_error": error,
                    "normalized_distance_to_interval": margin,
                    "interval_ok": interval_ok,
                }
            )
        rows.append({"spin": float(spin), "channels": channels})

    # Directly re-solve only the closest rows.  This guards the decimal-rounding
    # decision without regenerating the full fixed grid through the expensive
    # continued-fraction solver.
    standalone: List[Dict[str, Any]] = []
    selected_indices: List[int] = []
    for _margin, index, _channel in sorted(candidates)[:6]:
        if index not in selected_indices:
            selected_indices.append(index)
    try:
        from kerr_leaver_solver import solve
    except Exception as exc:  # pragma: no cover - optional diagnostic dependency
        standalone.append({"status": "UNAVAILABLE", "reason": str(exc)})
    else:
        for index in selected_indices:
            spin = float(KNOT_SPINS[index])
            try:
                root = solve(abs(spin), 2, 2 if spin >= 0.0 else -2, 0)
                actual = (
                    float(root.omega.real) / (2.0 * math.pi),
                    float(-root.omega.imag) / (2.0 * math.pi),
                )
                expected = tuple(
                    _round_significant(value, ROUNDING_SIG_DIGITS) for value in actual
                )
                target = ROUNDED_CYCLIC_ORDINATES[index]
                standalone.append(
                    {
                        "status": "PASSED" if expected == target else "FAILED",
                        "spin": spin,
                        "actual_cyclic": actual,
                        "expected_rounding": expected,
                        "table_row": target,
                        "radial_residual_abs": abs(root.radial_residual),
                        "angular_error": float(root.angular_error),
                    }
                )
                valid = valid and expected == target
            except Exception as exc:  # pragma: no cover - only near solver failures
                valid = False
                standalone.append({"status": "FAILED", "spin": spin, "reason": str(exc)})

    return {
        "status": "PASSED" if valid else "FAILED",
        "grid_rows": len(rows),
        "minimum_normalized_rounding_margin": minimum_margin,
        "rows": rows,
        "standalone_solver_checks": standalone,
    }


def _frozen_d_cases() -> List[Dict[str, Any]]:
    """Select the four pre-existing D gate cases without changing their gates."""

    # This wrapper supplies the existing public gate inventory.  The exact
    # expected_gate objects come from the canonical harness, so this script
    # never invents or loosens a threshold.
    from diagnose_d_qnm_contract import _load_gate_cases
    from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases

    required_ids = set(_load_gate_cases())
    by_id = {}
    for case in get_canonical_catalog_cases() + get_canonical_holdout_cases():
        if case.get("torch_name") == "IMRPhenomD" and case.get("id") in required_ids:
            by_id[case["id"]] = case
    cases = [by_id[key] for key in sorted(by_id)]
    if len(cases) != 4:
        raise RuntimeError(f"Expected four frozen D cases, found {len(cases)}")
    return cases


def _frozen_gate_report(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from parity_harness import run_oracle_comparison
    except Exception as exc:  # pragma: no cover - diagnostic-only dependency
        return [{"status": "UNAVAILABLE", "reason": str(exc)}]

    # Production IMRPhenomD now calls the sampler directly.  Keep this report
    # on that route: a temporary provider would hide import/call-site errors.
    return [run_oracle_comparison(case, strict=False) for case in cases]


def _frozen_nrtidal_nsbh_cases() -> List[Dict[str, Any]]:
    """Select the existing catalog and holdout NRTidal/NSBH gate cases."""

    from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases

    names = {"IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2", "IMRPhenomNSBH"}
    cases = [
        case
        for case in get_canonical_catalog_cases() + get_canonical_holdout_cases()
        if case.get("torch_name") in names
    ]
    return sorted(cases, key=lambda case: case["id"])


def _autograd_value_derivative(evaluator, spin_value: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Evaluate a scalar two-channel function and its spin derivative."""

    spin = torch.tensor(spin_value, dtype=torch.float64, requires_grad=True)
    values = tuple(evaluator(spin))
    derivatives = []
    for index, value in enumerate(values):
        derivatives.append(
            float(torch.autograd.grad(value, spin, retain_graph=index == 0)[0])
        )
    return tuple(float(value.detach()) for value in values), tuple(derivatives)


def _fallback_boundary_report() -> List[Dict[str, Any]]:
    """Quantify the one-sided value and derivative jump at each route join."""

    records = []
    for boundary, inner_side, outer_side in (
        (COMPATIBILITY_SPIN_MIN, "+epsilon", "-epsilon"),
        (COMPATIBILITY_SPIN_MAX, "-epsilon", "+epsilon"),
    ):
        table_values, table_derivatives = _autograd_value_derivative(
            evaluate_d_qnm_220, boundary
        )
        generated_values, generated_derivatives = _autograd_value_derivative(
            independent_cyclic_qnm_220, boundary
        )
        channels = []
        for channel, (table_value, generated_value, table_derivative, generated_derivative) in enumerate(
            zip(table_values, generated_values, table_derivatives, generated_derivatives)
        ):
            value_jump = generated_value - table_value
            derivative_jump = generated_derivative - table_derivative
            value_scale = max(abs(table_value), abs(generated_value), 1e-300)
            derivative_scale = max(abs(table_derivative), abs(generated_derivative), 1e-300)
            channels.append(
                {
                    "channel": "M_f_f_RD" if channel == 0 else "M_f_f_damp",
                    "table_limit": table_value,
                    "generated_limit": generated_value,
                    "value_jump_generated_minus_table": value_jump,
                    "relative_value_jump": abs(value_jump) / value_scale,
                    "table_derivative": table_derivative,
                    "generated_derivative": generated_derivative,
                    "derivative_jump_generated_minus_table": derivative_jump,
                    "relative_derivative_jump": abs(derivative_jump) / derivative_scale,
                }
            )
        records.append(
            {
                "boundary_spin": boundary,
                "table_side": inner_side,
                "generated_side": outer_side,
                "channels": channels,
            }
        )
    return records


def _inference_autograd_control() -> Dict[str, Any]:
    """Exercise production remnant inference first, then a differentiable call."""

    values = tuple(torch.tensor(value, dtype=torch.float64) for value in (30.0, 20.0, 0.3, -0.2))
    with torch.inference_mode():
        inferred = IMRPhenomD.remnant_properties(*values)

    differentiable = list(values)
    differentiable[2] = differentiable[2].detach().requires_grad_()
    remnant = IMRPhenomD.remnant_properties(*differentiable)
    gradient = torch.autograd.grad(remnant[2] + remnant[3], differentiable[2])[0]
    finite = bool(
        all(torch.isfinite(value).all() for value in inferred)
        and all(torch.isfinite(value).all() for value in remnant)
        and torch.isfinite(gradient).all()
    )
    return {
        "status": "PASSED" if finite else "FAILED",
        "validation_scope": "inference-first remnant call followed by autograd remnant call",
        "inference_outputs_finite": bool(all(torch.isfinite(value).all() for value in inferred)),
        "autograd_outputs_finite": bool(all(torch.isfinite(value).all() for value in remnant)),
        "autograd_gradient_finite": bool(torch.isfinite(gradient).all()),
    }


def _waveform_summary(hp: torch.Tensor, hc: torch.Tensor) -> Dict[str, Any]:
    finite = bool(torch.isfinite(hp.real).all() and torch.isfinite(hp.imag).all()
                  and torch.isfinite(hc.real).all() and torch.isfinite(hc.imag).all())
    return {
        "finite": finite,
        "shape_plus": list(hp.shape),
        "shape_cross": list(hc.shape),
        "dtype_plus": str(hp.dtype),
        "dtype_cross": str(hc.dtype),
        "max_abs_plus": float(torch.abs(hp).max()) if hp.numel() else 0.0,
        "max_abs_cross": float(torch.abs(hc).max()) if hc.numel() else 0.0,
    }


_CONTROL_CASES: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    (
        "IMRPhenomD_negative_final_spin",
        "IMRPhenomD",
        {
            "mass1": 80.0,
            "mass2": 8.0,
            "chi1": -0.7,
            "chi2": 0.1,
            "distance": 500.0,
            "inclination": 0.4,
            "phic": 0.2,
            "f_ref": 20.0,
        },
    ),
    (
        "IMRPhenomD_high_final_spin",
        "IMRPhenomD",
        {
            "mass1": 1000.0,
            "mass2": 1.0,
            "chi1": 1.0,
            "chi2": 1.0,
            "distance": 500.0,
            "inclination": 0.4,
            "phic": 0.2,
            "f_ref": 20.0,
        },
    ),
    (
        "IMRPhenomD_NRTidal",
        "IMRPhenomD_NRTidal",
        {
            "mass1": 1.5,
            "mass2": 1.3,
            "chi1": 0.02,
            "chi2": -0.03,
            "lambda1": 400.0,
            "lambda2": 800.0,
            "distance": 100.0,
            "inclination": 0.5,
            "phic": 0.3,
            "f_ref": 20.0,
        },
    ),
    (
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomD_NRTidalv2",
        {
            "mass1": 1.5,
            "mass2": 1.3,
            "chi1": 0.02,
            "chi2": -0.03,
            "lambda1": 400.0,
            "lambda2": 800.0,
            "distance": 100.0,
            "inclination": 0.5,
            "phic": 0.3,
            "f_ref": 20.0,
        },
    ),
    (
        "IMRPhenomNSBH",
        "IMRPhenomNSBH",
        {
            "mass1": 7.0,
            "mass2": 1.4,
            "chi1": 0.4,
            "chi2": 0.0,
            "lambda2": 2600.0,
            "distance": 100.0,
            "inclination": 0.5,
            "phic": 0.3,
            "f_ref": 20.0,
        },
    ),
    (
        "IMRPhenomHM",
        "IMRPhenomHM",
        {
            "mass1": 30.0,
            "mass2": 20.0,
            "chi1": 0.3,
            "chi2": -0.2,
            "distance": 400.0,
            "inclination": 0.5,
            "phic": 0.3,
            "f_ref": 20.0,
            "modes": [(2, 2)],
        },
    ),
    (
        "IMRPhenomPv2",
        "IMRPhenomPv2",
        {
            "mass1": 42.0,
            "mass2": 17.0,
            "chi1": -0.3,
            "chi2": 0.2,
            "spin1x": 0.3,
            "spin1y": -0.15,
            "spin1z": -0.3,
            "spin2x": 0.12,
            "spin2y": 0.08,
            "spin2z": 0.2,
            "distance": 680.0,
            "inclination": 1.1,
            "phic": 0.71,
            "f_ref": 35.0,
        },
    ),
)


def _control_report() -> List[Dict[str, Any]]:
    """Check finite production outputs; these controls have no oracle gate."""

    from torchwave.waveforms import get_fd_waveform

    frequency = torch.arange(20.0, 512.25, 0.25, dtype=torch.float64)
    records = []
    with torch.no_grad():
        for name, approximant, params in _CONTROL_CASES:
            try:
                waveform = get_fd_waveform(approximant, sample_frequencies=frequency, **params)
                output_summary = _waveform_summary(*waveform)
                remnant = IMRPhenomD.remnant_properties(
                    torch.tensor(params["mass1"], dtype=torch.float64),
                    torch.tensor(params["mass2"], dtype=torch.float64),
                    torch.tensor(params.get("chi1", params.get("spin1z", 0.0)), dtype=torch.float64),
                    torch.tensor(params.get("chi2", params.get("spin2z", 0.0)), dtype=torch.float64),
                )
                final_spin = float(remnant[1])
                records.append(
                    {
                        "name": name,
                        "status": "FINITE_OUTPUT_ONLY" if output_summary["finite"] else "FAILED",
                        "validation_scope": "finite output, shape, and dtype only; no oracle parity gate",
                        "final_spin": final_spin,
                        "path": "rounded_cubic" if abs(final_spin) <= COMPATIBILITY_SPIN_MAX else "generated_fallback",
                        "output": output_summary,
                    }
                )
            except Exception as exc:
                records.append({"name": name, "status": "FAILED", "reason": str(exc)})
    return records


def main() -> None:
    cases = _frozen_d_cases()
    nrtidal_nsbh_cases = _frozen_nrtidal_nsbh_cases()
    rounding = _rounding_report()
    frozen = _frozen_gate_report(cases)
    frozen_nrtidal_nsbh = _frozen_gate_report(nrtidal_nsbh_cases)
    fallback_boundary = _fallback_boundary_report()
    inference_autograd = _inference_autograd_control()
    controls = _control_report()

    source_paths = (
        "scripts/diagnose_d_sampling_clean.py",
        "scripts/generate_kerr_qnm_independent.py",
        "scripts/kerr_leaver_solver.py",
        "src/torchwave/approximants/kerr_qnm_d_sampling_clean.py",
        "src/torchwave/approximants/kerr_qnm_generated.py",
    )
    versions = {}
    for package in ("torch", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"

    artifact = {
        "metadata": {
            "python": "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python",
            "versions": versions,
            "knot_count": N_KNOTS,
            "ordinary_grid": "a=k/500 for k=-499..499, with independently generated guards a=-.999 and a=.999",
            "rounded_units": "cyclic M_f f; independent angular roots divided by 2*pi before six-significant-digit rounding",
            "rounding_significant_digits": ROUNDING_SIG_DIGITS,
            "compatibility_domain": [COMPATIBILITY_SPIN_MIN, COMPATIBILITY_SPIN_MAX],
            "generated_domain": [-GENERATED_SPIN_LIMIT, GENERATED_SPIN_LIMIT],
            "endpoint_policy": ENDPOINT_POLICY,
            "source_sha256": {path: _sha256(path) for path in source_paths},
        },
        "rounding": rounding,
        "frozen_d_standard_gates": frozen,
        "frozen_nrtidal_nsbh_gates": frozen_nrtidal_nsbh,
        "fallback_boundary": fallback_boundary,
        "inference_autograd_control": inference_autograd,
        "controls": controls,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "artifact": str(ARTIFACT.relative_to(ROOT)),
        "rounding": rounding["status"],
        "frozen_gate_statuses": [row.get("status") for row in frozen],
        "frozen_nrtidal_nsbh_gate_statuses": [row.get("status") for row in frozen_nrtidal_nsbh],
        "fallback_boundary": fallback_boundary,
        "inference_autograd": inference_autograd["status"],
        "control_statuses": {row["name"]: row["status"] for row in controls},
    }, indent=2))


if __name__ == "__main__":
    main()
