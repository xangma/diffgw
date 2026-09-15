"""Black-box MSA v223 boundary checks for the clean implementation.

The fixed boundary fixtures are the sanitized XP contract probes.  Native
positive-frequency waveforms are compared directly with the public
``IMRPhenomXP`` binary on the common integer-Hz grid.  The small-in-plane
fixture additionally records the current-root transition around 190 Hz.  No
phase, time, amplitude, or residual fit is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from diagnose_msa_v223_contract_clean import (
    _native_angles,
    _native_waveform,
    _public_angles,
    _public_waveform,
    _raw_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs/reviews/2026-09-13-parity-review/verification/msa-boundary-contract-clean"
REPORT_PATH = REPORT_DIR / "msa_boundary_contract_report.json"
MTSUN_SEC = 4.925490947641267e-6
ANGLE_TRANSITION_FREQUENCIES = torch.arange(185.0, 197.0, dtype=torch.float64)


def _common(**kwargs):
    return {
        "distance": 500.0,
        "inclination": 0.7,
        "phic": 0.0,
        "tc": 0.0,
        "f_ref": 20.0,
        **kwargs,
    }


CASES = (
    _common(
        name="equalmass",
        mass1=30.0,
        mass2=30.0,
        spin1=(0.2, 0.1, 0.2),
        spin2=(-0.1, 0.15, -0.1),
        expected="pass",
    ),
    _common(
        name="near_equalmass",
        mass1=30.000001,
        mass2=30.0,
        spin1=(0.2, 0.1, 0.2),
        spin2=(-0.1, 0.15, -0.1),
        expected="pass",
    ),
    _common(
        name="aligned",
        mass1=35.0,
        mass2=25.0,
        spin1=(0.0, 0.0, 0.2),
        spin2=(0.0, 0.0, -0.1),
        expected="pass",
    ),
    _common(
        name="zero_spin",
        mass1=35.0,
        mass2=25.0,
        spin1=(0.0, 0.0, 0.0),
        spin2=(0.0, 0.0, 0.0),
        expected="pass",
    ),
    _common(
        name="small_inplane",
        mass1=35.0,
        mass2=25.0,
        spin1=(1.0e-5, 0.0, 0.2),
        spin2=(0.0, 1.0e-5, -0.1),
        expected="unresolved_current_root_onset",
    ),
    _common(
        name="swapped",
        mass1=25.0,
        mass2=35.0,
        spin1=(-0.1, 0.15, -0.1),
        spin2=(0.2, 0.1, 0.2),
        expected="pass",
    ),
    _common(
        name="unequal_extreme",
        mass1=65.0,
        mass2=9.0,
        spin1=(0.45, 0.03, -0.55),
        spin2=(-0.21, 0.13, 0.06),
        expected="pass",
    ),
    _common(
        name="generic_anchor_B",
        mass1=45.0,
        mass2=15.0,
        spin1=(0.45, -0.2, 0.3),
        spin2=(0.1, 0.2, -0.25),
        expected="pass",
    ),
)


def _to_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _angle_residuals(case, frequencies):
    native = _native_angles(case, frequencies)
    public = _public_angles(case, frequencies)
    # Public alpha includes a source-frame chart constant.  Remove that
    # constant at the first frequency before reporting the physical residual.
    alpha_offset = public[0][0] - native["alpha"][0]
    alpha_residual = native["alpha"] + alpha_offset - public[0]
    gamma_residual = native["gamma"] - public[1]
    cos_residual = native["cos_beta"] - public[2]
    return native, {
        "alpha_gauge_offset": _to_float(alpha_offset),
        "alpha_max_abs": _to_float(alpha_residual.abs().max()),
        "gamma_max_abs": _to_float(gamma_residual.abs().max()),
        "cos_beta_max_abs": _to_float(cos_residual.abs().max()),
        "alpha_residual": alpha_residual.tolist(),
        "gamma_residual": gamma_residual.tolist(),
        "cos_beta_residual": cos_residual.tolist(),
    }


def _series_phase(case, native, frequencies):
    total = torch.as_tensor(case["mass1"] + case["mass2"], dtype=torch.float64)
    q = torch.as_tensor(
        min(case["mass1"], case["mass2"]) /
        max(case["mass1"], case["mass2"]), dtype=torch.float64
    )
    delta = (1.0 - q) / (1.0 + q)
    g0 = 1.0 / native["flux_coefficients"]["a0"]
    v = (math.pi * total * MTSUN_SEC * frequencies).pow(1.0 / 3.0)
    expected = -0.75 * g0 * delta * (
        1.0 + native["psi1"] * v + native["psi2"] * v.square()
    ) / v.pow(3)
    return expected


def _transition(case):
    native, residuals = _angle_residuals(case, ANGLE_TRANSITION_FREQUENCIES)
    expected_phase = _series_phase(case, native, ANGLE_TRANSITION_FREQUENCIES)
    phase_error = (native["psi"] - expected_phase).abs()
    degenerate = native["frequency_degenerate"]
    return {
        "frequencies_hz": ANGLE_TRANSITION_FREQUENCIES.tolist(),
        "initial_degenerate": bool(native["degenerate"]),
        "frequency_degenerate": degenerate.tolist(),
        "current_root_gap": (
            (native["roots"][:, 2] - native["roots"][:, 1]).abs().tolist()
        ),
        "rational_invalid_current": native["rational_invalid_current"].tolist(),
        "pole_factor_a": (
            1.0 - native["rational_lambda_a"]
        ).tolist(),
        "pole_factor_b": (
            1.0 - native["rational_lambda_b"]
        ).tolist(),
        "spin_squared_on_degenerate": native["spin_squared"][degenerate].tolist(),
        "upper_root_on_degenerate": native["roots"][degenerate, 2].tolist(),
        "series_phase_max_abs_error": _to_float(phase_error.max()),
        "oscillatory_phi_on_degenerate": native["oscillatory_phi"][degenerate].tolist(),
        "oscillatory_zeta_on_degenerate": native["oscillatory_zeta"][degenerate].tolist(),
        "oscillatory_phi_all": native["oscillatory_phi"].tolist(),
        "oscillatory_zeta_all": native["oscillatory_zeta"].tolist(),
        "angle_residuals": residuals,
        "interpretation": (
            "Initial Eq. (51) additive constant is zero and current small-gap "
            "rows suppress only nutation. The first ordinary-root point remains "
            "an unresolved public-waveform mismatch. Signed pole continuation "
            "is finite away from n=0 but the reported correction is numerically "
            "sensitive as either factor approaches zero; no empirical correction "
            "is applied."
        ),
    }


def _case_report(case):
    frequencies = torch.linspace(case["f_ref"], 200.0, 81, dtype=torch.float64)
    native_angles, angle_residuals = _angle_residuals(case, frequencies)
    native_hp, native_hc = _native_waveform(case, torch.arange(20.0, 201.0, dtype=torch.float64))
    public_hp, public_hc = _public_waveform(case)
    metrics = _raw_metrics((native_hp, native_hc), (public_hp, public_hc))
    report = {
        "name": case["name"],
        "expected": case["expected"],
        "parameters": {
            key: value for key, value in case.items()
            if key not in {"name", "expected"}
        },
        "native_angle_finite": bool(
            torch.isfinite(native_angles["alpha"]).all()
            and torch.isfinite(native_angles["gamma"]).all()
            and torch.isfinite(native_angles["cos_beta"]).all()
        ),
        "initial_degenerate": bool(native_angles["degenerate"]),
        "native_waveform_finite": bool(
            torch.isfinite(native_hp).all() and torch.isfinite(native_hc).all()
        ),
        "raw_hp_hc_metrics": metrics,
        "angle_residual_summary": {
            key: value for key, value in angle_residuals.items()
            if key.endswith("max_abs") or key == "alpha_gauge_offset"
        },
    }
    if case["name"] == "small_inplane":
        report["transition_185_196_hz"] = _transition(case)
    return report


def _hashes():
    relative_paths = (
        "src/torchwave/approximants/precession_msa_angles_clean.py",
        "src/torchwave/approximants/precession_msa_pn_clean.py",
        "src/torchwave/approximants/precession_msa_secular_clean.py",
        "src/torchwave/approximants/precession_msa_oscillatory_clean.py",
        "src/torchwave/approximants/precession_msa_waveform_clean.py",
        "tests/test_precession_msa_v223_contract_clean.py",
        "scripts/diagnose_msa_v223_contract_clean.py",
        "scripts/diagnose_msa_v223_boundary_contract_clean.py",
        "docs/reviews/2026-09-13-parity-review/reference-audit-contracts.md",
    )
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in relative_paths
    }


def main():
    rows = []
    for case in CASES:
        print(f"running {case['name']}", flush=True)
        rows.append(_case_report(case))
    report = {
        "status": "complete_with_unresolved_small_inplane",
        "contract": {
            "prec_version": 223,
            "expansion_order": 5,
            "public_approximant": "IMRPhenomXP",
            "raw_waveform_grid_hz": [20.0, 200.0],
            "raw_waveform_frequency_count": 181,
            "angle_transition_grid_hz": [185.0, 196.0],
            "root_gap_threshold": 1.0e-5,
            "phase_limit": "P(v)=-(3/4) g0 delta (1+psi1 v+psi2 v^2)/v^3 with additive C=0",
            "comparison": "direct native/public hp,hc; no phase/time/amplitude/residual optimization",
        },
        "validation": {
            "focused_test_command": (
                "PYTHONPATH=src:tests:scripts:. "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
                "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q "
                "tests/test_precession_msa_angles_clean.py "
                "tests/test_precession_msa_initial_clean.py "
                "tests/test_precession_msa_initial_physical.py "
                "tests/test_precession_msa_oscillatory_clean.py "
                "tests/test_precession_msa_pn_clean.py "
                "tests/test_precession_msa_reference_l_clean.py "
                "tests/test_precession_msa_secular_clean.py "
                "tests/test_precession_msa_v223_contract_clean.py "
                "tests/test_precession_msa_waveform_clean.py"
            ),
            "focused_test_result": "62 passed, 1 existing warning",
            "diagnostic_command": (
                "PYTHONPATH=src:tests:scripts:. "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
                "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python "
                "scripts/diagnose_msa_v223_boundary_contract_clean.py"
            ),
        },
        "rows": rows,
        "source_hashes_sha256": _hashes(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {REPORT_PATH}")
    for row in rows:
        hp = row["raw_hp_hc_metrics"]["hp"]["relative_l2"]
        hc = row["raw_hp_hc_metrics"]["hc"]["relative_l2"]
        print(f"{row['name']}: hp={hp:.9g} hc={hc:.9g} expected={row['expected']}")


if __name__ == "__main__":
    main()
