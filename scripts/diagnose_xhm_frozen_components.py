#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decompose frozen IMRPhenomXHM discrepancies without fitting the oracle.

The public LAL binary is called only through
``SimIMRPhenomXHMFrequencySequenceOneMode``.  Its outputs are never used to
derive production coefficients.  Amplitude regions use the current native
amplitude cuts; phase regions use the current native phase cuts.  Those cuts
are diagnostic labels, not independent certification of their fit surfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM, _eval_f_isco
from torchwave.approximants.xhm_fits import XHMParameterFits


MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    mass1: float
    mass2: float
    chi1: float
    chi2: float
    distance: float
    inclination: float
    phic: float
    tc: float
    f_ref: float


CASES = (
    FrozenCase("IMRPhenomXHM_diagnostic", 40.0, 20.0, 0.3, -0.1, 500.0, 1.0, 0.5, 0.0, 20.0),
    FrozenCase("IMRPhenomXHM_holdout_6", 55.0, 11.0, -0.4, 0.2, 680.0, 1.1, 0.71, 0.0, 27.0),
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _scalar(value: object) -> float:
    return float(torch.as_tensor(value, dtype=torch.float64).reshape(()))


def _public_mode(frequencies: np.ndarray, case: FrozenCase, mode: tuple[int, int]) -> np.ndarray:
    import lal
    import lalsimulation as ls

    vector = lal.CreateREAL8Vector(len(frequencies))
    vector.data = np.asarray(frequencies, dtype=np.float64)
    params = lal.CreateDict()
    series = ls.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector,
        case.mass1 * lal.MSUN_SI,
        case.mass2 * lal.MSUN_SI,
        case.chi1,
        case.chi2,
        mode[0],
        -mode[1],
        case.distance * 1.0e6 * lal.PC_SI,
        case.phic,
        case.f_ref,
        params,
    )
    return np.asarray(series.data.data, dtype=np.complex128).copy()


def _native_mode(frequencies: np.ndarray, case: FrozenCase, mode: tuple[int, int]) -> np.ndarray:
    with torch.no_grad():
        result = IMRPhenomXHM(modes=(mode,))(
            torch.as_tensor(frequencies, dtype=torch.float64),
            case.mass1,
            case.mass2,
            case.chi1,
            case.chi2,
            case.distance,
            case.inclination,
            case.phic,
            case.tc,
            case.f_ref,
            return_unprojected_modes=True,
        )
    return result[2][mode].detach().cpu().numpy().astype(np.complex128)


def _native_cuts_hz(case: FrozenCase, mode: tuple[int, int]) -> dict[str, list[float]]:
    """Return the production amplitude and phase cuts for one mode."""

    if mode == (2, 2):
        return {"amplitude": [], "phase": []}

    import lal

    m1, m2, s1, s2 = case.mass1, case.mass2, case.chi1, case.chi2
    total = m1 + m2
    eta = torch.tensor(m1 * m2 / total**2, dtype=torch.float64)
    spin1 = torch.tensor(s1, dtype=torch.float64)
    spin2 = torch.tensor(s2, dtype=torch.float64)
    chi_eff = (m1 * s1 + m2 * s2) / total
    final_spin = XASParameterFits.fit_final_spin(eta, spin1, spin2)
    final_mass = 1.0 - XASParameterFits.fit_erad(eta, spin1, spin2)
    f_meco = XASParameterFits.fit_f_meco(eta, spin1, spin2)
    delta_isco = torch.abs(_eval_f_isco(final_spin) - f_meco)
    f_ring = {
        key: getattr(XHMParameterFits, f"qnm_fring_{key}")(final_spin) / final_mass
        for key in ("21", "22", "32", "33", "44")
    }
    f_damp = {
        key: getattr(XHMParameterFits, f"qnm_fdamp_{key}")(final_spin) / final_mass
        for key in ("21", "22", "32", "33", "44")
    }
    l, m = mode
    key = f"{l}{m}"
    epsilon = {
        "21": torch.tensor(1.0),
        "33": torch.tensor(1.0),
        "32": f_ring["32"] / f_ring["22"],
        "44": torch.tensor(1.0),
    }[key]
    delta = {
        "21": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff**2,
        "33": 0.75 - 0.235 * chi_eff - (5.0 / 6.0) * chi_eff,
        "32": (0.75 - 0.235 * abs(chi_eff)) * f_ring["32"] / f_ring["22"],
        "44": 0.75 - 0.235 * chi_eff,
    }[key]
    f_ins_amp = (m / 2.0) * (f_meco * epsilon + delta_isco * delta)
    f_rd_amp = {
        "21": 0.75 * f_ring["21"],
        "33": 0.95 * f_ring["33"],
        "32": f_ring["22"],
        "44": 0.90 * f_ring["44"],
    }[key]
    f_ins_phase = (m / 2.0) * f_meco
    f_rd_phase = {
        "21": f_ring["21"] - f_damp["21"],
        "33": f_ring["33"] - f_damp["33"],
        "32": f_ring["22"] - 0.5 * f_damp["22"],
        "44": f_ring["44"] - f_damp["44"],
    }[key]
    mass_seconds = total * lal.MTSUN_SI
    return {
        "amplitude": [_scalar(f_ins_amp) / mass_seconds, _scalar(f_rd_amp) / mass_seconds],
        "phase": [_scalar(f_ins_phase) / mass_seconds, _scalar(f_rd_phase) / mass_seconds],
    }


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def _metrics(
    frequencies: np.ndarray,
    native: np.ndarray,
    oracle: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    f = frequencies[mask]
    native = native[mask]
    oracle = oracle[mask]
    if not len(f):
        raise ValueError("metric region is empty")
    amplitude_l2 = _relative_l2(np.abs(native), np.abs(oracle))
    native_norm_squared = np.vdot(native, native)
    if abs(native_norm_squared) <= np.finfo(float).tiny:
        raise ValueError("native mode norm is zero")
    scale = np.vdot(native, oracle) / native_norm_squared
    shape_l2 = _relative_l2(scale * native, oracle)

    valid = (
        (np.abs(native) > np.max(np.abs(native)) * 1.0e-10)
        & (np.abs(oracle) > np.max(np.abs(oracle)) * 1.0e-10)
    )
    f_phase = f[valid]
    ratio_phase = np.unwrap(np.angle(native[valid] / oracle[valid]))
    weights = np.abs(oracle[valid]) ** 2
    design = np.column_stack((np.ones(len(f_phase)), f_phase))
    normal = (design.T * weights) @ design
    rhs = (design.T * weights) @ ratio_phase
    offset, slope = np.linalg.solve(normal, rhs)
    residual = ratio_phase - design @ np.asarray((offset, slope))
    centered = ratio_phase - np.average(ratio_phase, weights=weights)
    return {
        "sample_count": int(len(f)),
        "phase_sample_count": int(len(f_phase)),
        "amplitude_relative_l2": amplitude_l2,
        "complex_relative_l2_after_optimal_scalar": shape_l2,
        "phase_rms_after_constant_rad": float(np.sqrt(np.average(centered**2, weights=weights))),
        "phase_rms_after_affine_rad": float(np.sqrt(np.average(residual**2, weights=weights))),
        "phase_affine_offset_rad": float(offset),
        "phase_affine_slope_rad_per_hz": float(slope),
    }


def _regions(lower: float, upper: float, cuts: list[float]) -> list[tuple[str, float, float]]:
    points = sorted(set([lower, upper, *(min(upper, max(lower, x)) for x in cuts)]))
    names = ("inspiral", "intermediate", "ringdown") if cuts else ("all",)
    intervals = [(a, b) for a, b in zip(points[:-1], points[1:]) if b > a]
    return [(names[min(index, len(names) - 1)], a, b) for index, (a, b) in enumerate(intervals)]


def _git_revision() -> str:
    supplied = os.environ.get("DIFFGW_GIT_REVISION")
    if supplied:
        return supplied
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def diagnose() -> dict[str, object]:
    import lal
    import lalsimulation as ls

    frequencies = np.arange(20.0, 512.0 + 0.125, 0.25, dtype=np.float64)
    cases: dict[str, object] = {}
    for case in CASES:
        mode_rows: dict[str, object] = {}
        for mode in MODES:
            oracle = _public_mode(frequencies, case, mode)
            native = _native_mode(frequencies, case, mode)
            cuts = _native_cuts_hz(case, mode)
            row: dict[str, object] = {
                "native_transition_frequencies_hz": cuts,
                "all": _metrics(frequencies, native, oracle, np.ones_like(frequencies, dtype=bool)),
            }
            for quantity in ("amplitude", "phase"):
                row[f"{quantity}_regions"] = {
                    name: {
                        "frequency_hz": [lo, hi],
                        "metrics": _metrics(
                            frequencies,
                            native,
                            oracle,
                            (frequencies >= lo) & (frequencies <= hi if hi == 512.0 else frequencies < hi),
                        ),
                    }
                    for name, lo, hi in _regions(20.0, 512.0, cuts[quantity])
                }
            mode_rows[f"{mode[0]}{mode[1]}"] = row
        cases[case.case_id] = {"parameters": asdict(case), "modes": mode_rows}

    return {
        "schema": "diffgw.xhm.frozen-component-diagnostic/v1",
        "evidence_category": "external-binary-numerical-oracle",
        "production_fit_policy": "oracle outputs are diagnostic only and are not fitted or transcribed",
        "frequency_grid_hz": {"start": 20.0, "stop": 512.0, "delta": 0.25},
        "metrics": {
            "amplitude": "relative L2 of magnitudes",
            "complex_shape": "complex relative L2 after one optimal complex scalar",
            "phase": "oracle-amplitude-weighted residual after nuisance constant or affine phase alignment",
        },
        "versions": {
            "diffgw_git_revision": _git_revision(),
            "python_lal": getattr(lal, "__version__", _package_version("lalsuite")),
            "python_lalsimulation": getattr(ls, "__version__", _package_version("lalsuite")),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    rendered = json.dumps(diagnose(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
