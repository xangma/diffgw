"""Development diagnostic of the native conditioned FD waveform against public SEOBNRv4.

The free complex phase and lag fit is descriptive, not a physical overlap or
release gate. Public LAL is used only as an external oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from . import qualify_generator as q
from .eob.torch.conditioned_waveform_clean import conditioned_seobnrv4_fd


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/native_public_fd_plan.json"
FROZEN_ROWS = [
    dict(name="mixed_spin", mass1=30.0, mass2=20.0, chi1z=0.3, chi2z=-0.2,
         f_low=20.0),
    dict(name="safe_frequency_cap", mass1=50.0, mass2=30.0, chi1z=0.8,
         chi2z=0.5, f_low=60.0),
    dict(name="q4_nonspinning", mass1=40.0, mass2=10.0, chi1z=0.0,
         chi2z=0.0, f_low=20.0),
    dict(name="strong_antialigned", mass1=30.0, mass2=20.0, chi1z=-0.8,
         chi2z=-0.6, f_low=20.0),
]


def _public_fd(row: dict, plan: dict):
    return lalsim.SimInspiralFD(
        row["mass1"] * lal.MSUN_SI, row["mass2"] * lal.MSUN_SI,
        0.0, 0.0, row["chi1z"], 0.0, 0.0, row["chi2z"],
        plan["distance_mpc"] * 1e6 * lal.PC_SI,
        plan["inclination_rad"], plan["reference_phase_rad"],
        0.0, 0.0, 0.0, plan["delta_f_hz"], row["f_low"],
        plan["f_max_hz"], plan["reference_frequency_hz"],
        lal.CreateDict(), lalsim.SEOBNRv4,
    )


def _row(row: dict, plan: dict) -> dict:
    start = time.monotonic()
    native = conditioned_seobnrv4_fd(
        row["mass1"], row["mass2"], chi1z=row["chi1z"], chi2z=row["chi2z"],
        f_low=row["f_low"], f_max=plan["f_max_hz"], delta_f=plan["delta_f_hz"],
        distance=plan["distance_mpc"], inclination=plan["inclination_rad"],
        phic=plan["reference_phase_rad"], return_metadata=True,
    )
    native_seconds = time.monotonic() - start
    start = time.monotonic()
    public = _public_fd(row, plan)
    public_seconds = time.monotonic() - start
    native_arrays = [x.detach().cpu().numpy() for x in (native.h_plus, native.h_cross)]
    public_arrays = [np.asarray(x.data.data).copy() for x in public]
    if any(a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
           for a, b in zip(native_arrays, public_arrays)):
        raise ValueError("native/public complex FD grids or finiteness differ")
    if (any(x.deltaF != plan["delta_f_hz"] for x in public)
            or len(native.frequencies) != len(public_arrays[0])):
        raise ValueError("native/public frequency grids differ")
    f = np.arange(len(public_arrays[0]), dtype=np.float64) * plan["delta_f_hz"]
    mask = (f >= row["f_low"]) & (f <= plan["f_max_hz"])
    frequency = f[mask]
    a = np.concatenate([x[mask] for x in native_arrays])
    b = np.concatenate([x[mask] for x in public_arrays])
    doubled_frequency = np.concatenate([frequency, frequency])
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("zero native/public waveform norm")

    def cross_at_lag(lag: float) -> complex:
        return np.vdot(a * np.exp(-2j * np.pi * doubled_frequency * lag), b)

    bound = plan["fit_lag_bound_seconds"]
    grid = np.linspace(-bound, bound, plan["fit_lag_grid_samples"])
    strengths = np.array([abs(cross_at_lag(float(lag))) for lag in grid])
    peak = int(np.argmax(strengths))
    lower, upper = grid[max(0, peak - 1)], grid[min(len(grid) - 1, peak + 1)]
    optimum = minimize_scalar(
        lambda lag: -abs(cross_at_lag(lag)), bounds=(lower, upper),
        method="bounded", options={"xatol": 1e-12},
    )
    lag = float(optimum.x)
    cross = cross_at_lag(lag)
    coherence = abs(cross) / (norm_a * norm_b)
    fitted_relative = float(np.sqrt(max(0.0, 1.0 - coherence**2)))
    aligned = a * np.exp(-2j * np.pi * doubled_frequency * lag) * cross / norm_a**2

    # For h_2,-2 = conj(h_22), the projection has h+ proportional to
    # (p+m) Re(h22) and hx proportional to -(p-m) Im(h22), with
    # p=(1+cos(i))^2 and m=(1-cos(i))^2. An orbital-phase rotation therefore
    # mixes the two polarizations with these unequal weights.
    cosine = math.cos(plan["inclination_rad"])
    plus_over_cross = (1.0 + cosine**2) / (2.0 * cosine)
    plus_native, cross_native = (x[mask] for x in native_arrays)

    def physical_residual(parameters: np.ndarray) -> float:
        lag_seconds, phase = float(parameters[0]) * 1e-3, float(parameters[1])
        c, s = math.cos(phase), math.sin(phase)
        shifted = np.exp(-2j * np.pi * frequency * lag_seconds)
        rotated_plus = (c * plus_native - s * plus_over_cross * cross_native) * shifted
        rotated_cross = (c * cross_native + s / plus_over_cross * plus_native) * shifted
        return float((np.linalg.norm(rotated_plus - public_arrays[0][mask])**2
                      + np.linalg.norm(rotated_cross - public_arrays[1][mask])**2)
                     / norm_b**2)

    fit_seed = np.array([lag * 1e3, np.angle(cross)])
    physical_fit = minimize(
        physical_residual, x0=fit_seed,
        method="Nelder-Mead",
        bounds=[(-bound * 1e3, bound * 1e3), (-math.pi, math.pi)],
        options={
            "initial_simplex": np.array([
                fit_seed, fit_seed + [0.05, 0.0], fit_seed + [0.0, 0.01],
            ]),
            "xatol": 1e-10, "fatol": 1e-15, "maxiter": 400,
        },
    )
    raw_error = float(np.linalg.norm(a - b) / norm_b)
    if not physical_fit.success or physical_fit.fun > raw_error**2 + 1e-14:
        raise ValueError(f"physical phase/time fit did not converge: {physical_fit.message}")
    band_errors = []
    for index, (low, high) in enumerate(plan["bands_hz"]):
        in_band = (doubled_frequency >= max(low, row["f_low"])) & (
            doubled_frequency <= high if index == len(plan["bands_hz"]) - 1
            else doubled_frequency < high
        )
        band_norm = np.linalg.norm(b[in_band])
        band_errors.append({
            "band_hz": [low, high],
            "public_norm_fraction": float(band_norm / norm_b),
            "fitted_complex_relative_l2": float(
                np.linalg.norm(aligned[in_band] - b[in_band]) / band_norm
            ) if band_norm else None,
            "amplitude_relative_l2": float(
                np.linalg.norm(abs(a[in_band]) - abs(b[in_band])) / band_norm
            ) if band_norm else None,
        })
    nfft = native.sampling.n_time
    conditioned_length = len(native.conditioning.h_plus)
    padded_epoch = float(native.physical_epoch) - (nfft - conditioned_length) * float(native.dt_seconds)
    return {
        "name": row["name"], "native_seconds": native_seconds,
        "public_seconds": public_seconds,
        "native_td_samples": len(native.td_time),
        "native_conditioned_samples": conditioned_length,
        "public_fd_samples": len(public_arrays[0]),
        "native_fd_sha256": hashlib.sha256(
            native_arrays[0].tobytes() + native_arrays[1].tobytes()).hexdigest(),
        "public_fd_sha256": hashlib.sha256(
            public_arrays[0].tobytes() + public_arrays[1].tobytes()).hexdigest(),
        "native_padded_epoch_seconds": padded_epoch,
        "public_epoch_seconds": float(public[0].epoch),
        "raw_complex_relative_l2": raw_error,
        "amplitude_relative_l2": float(np.linalg.norm(abs(a) - abs(b)) / norm_b),
        "native_to_public_norm_ratio": float(norm_a / norm_b),
        "fitted_lag_seconds": lag,
        "fitted_free_complex_phase_rad": float(np.angle(cross_at_lag(lag))),
        "fitted_complex_coherence": float(coherence),
        "fitted_relative_l2_with_free_complex_scale": fitted_relative,
        "fitted_residual_by_frequency_band": band_errors,
        "physical_fixed_distance_fit": {
            "lag_seconds": float(physical_fit.x[0]) * 1e-3,
            "orbital_phase_rotation_rad": float(physical_fit.x[1]) / 2.0,
            "relative_l2": float(np.sqrt(physical_fit.fun)),
            "lag_at_bound": bool(abs(physical_fit.x[0]) >= bound * 1e3 - 1e-7),
        },
        "lag_fit_at_bound": bool(peak in (0, len(grid) - 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.native-public-fd-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != FROZEN_ROWS
            or plan["distance_mpc"] != 1.0
            or plan["inclination_rad"] != 0.7
            or plan["reference_phase_rad"] != 0.0
            or plan["reference_frequency_hz"] != 0.0
            or plan["f_max_hz"] != 2048.0
            or plan["delta_f_hz"] != 0.25
            or plan["fit_lag_bound_seconds"] != 0.05
            or plan["fit_lag_grid_samples"] != 501
            or plan["physical_fit"] != "common lag and quadrupole polarization rotation at fixed distance and inclination; bounded Nelder-Mead local refinement from the free-complex-fit seed"
            or plan["bands_hz"] != [[20.0, 100.0], [100.0, 300.0],
                                    [300.0, 700.0], [700.0, 2048.0]]):
        parser.error("frozen native/public FD plan differs")
    evidence = q.ROOT / plan["binary_evidence"]
    if q.sha256(evidence) != plan["binary_evidence_sha256"]:
        parser.error("pinned binary evidence differs")
    frozen_binary = json.loads(evidence.read_text())["lal_binary"]
    binary = q.lal_binary_identity(
        frozen_binary["lal_version"], frozen_binary["lalsimulation_version"]
    )
    if binary != frozen_binary:
        parser.error("installed public binary differs from frozen oracle")
    report = {
        "schema": "diffgw.seobnrv4.native-public-fd.v1",
        "status": "running", "release_qualified": False,
        "scope": "Selected native generator plus conditioning versus pinned public complex FD; free complex phase/lag fit is diagnostic only.",
        "known_limitations": [
            "The free complex phase is not a physical polarization rotation or detector-weighted overlap.",
            "Four selected cases cannot qualify the parameter and frequency domain.",
            "Native source fidelity, physical NR validation, full gradients, and ROM training remain open.",
        ],
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "lal_binary": binary,
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    for row in plan["rows"]:
        try:
            result = _row(row, plan)
            report["rows"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            report["errors"].append({"name": row["name"], "error": repr(exc)})
            print(f"{row['name']}: {exc!r}", flush=True)
    report["status"] = "measured" if not report["errors"] else "failed"
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
