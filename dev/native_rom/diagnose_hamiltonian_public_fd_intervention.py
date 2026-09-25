"""Dev-only FD intervention using the pinned source's Hamiltonian choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import minimize

from . import qualify_generator as q
from .diagnose_native_public_fd import FROZEN_ROWS, _public_fd
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .eob.torch.conditioned_waveform_clean import conditioned_seobnrv4_fd


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/hamiltonian_public_fd_intervention_plan.json"


def _intervention_generator(*args, **kwargs):
    return reconstruct_seobnrv4_22_adaptive(
        *args, **kwargs,
        diagnostic_hamiltonian_log_derivative="lal_v4",
        diagnostic_hamiltonian_calibration="lal_v4_rounded",
    )


def _row(row: dict, baseline: dict, fd_plan: dict) -> dict:
    start = time.monotonic()
    native = conditioned_seobnrv4_fd(
        row["mass1"], row["mass2"], chi1z=row["chi1z"], chi2z=row["chi2z"],
        f_low=row["f_low"], f_max=fd_plan["f_max_hz"],
        delta_f=fd_plan["delta_f_hz"], distance=fd_plan["distance_mpc"],
        inclination=fd_plan["inclination_rad"],
        phic=fd_plan["reference_phase_rad"],
        generator=_intervention_generator, return_metadata=True,
    )
    generation_seconds = time.monotonic() - start
    public = _public_fd(row, fd_plan)
    native_arrays = [np.asarray(x.detach().cpu().numpy())
                     for x in (native.h_plus, native.h_cross)]
    public_arrays = [np.asarray(x.data.data).copy() for x in public]
    public_sha = hashlib.sha256(
        public_arrays[0].tobytes() + public_arrays[1].tobytes()).hexdigest()
    if public_sha != baseline["public_fd_sha256"]:
        raise ValueError("public FD array differs from the frozen baseline")
    if any(a.shape != b.shape or not np.isfinite(a).all()
           for a, b in zip(native_arrays, public_arrays)):
        raise ValueError("native/public FD grid or finiteness differs")
    frequencies = np.arange(len(public_arrays[0])) * fd_plan["delta_f_hz"]
    mask = (frequencies >= row["f_low"]) & (frequencies <= fd_plan["f_max_hz"])
    f = frequencies[mask]
    a = np.concatenate([x[mask] for x in native_arrays])
    b = np.concatenate([x[mask] for x in public_arrays])
    norm_b = np.linalg.norm(b)
    raw = float(np.linalg.norm(a - b) / norm_b)
    cosine = math.cos(fd_plan["inclination_rad"])
    ratio = (1 + cosine**2) / (2 * cosine)
    plus, cross = (x[mask] for x in native_arrays)
    public_plus, public_cross = (x[mask] for x in public_arrays)

    def residual(parameters):
        lag, phase = float(parameters[0]) * 1e-3, float(parameters[1])
        shifted = np.exp(-2j * np.pi * f * lag)
        c, s = math.cos(phase), math.sin(phase)
        rotated_plus = (c * plus - s * ratio * cross) * shifted
        rotated_cross = (c * cross + s / ratio * plus) * shifted
        return float((np.linalg.norm(rotated_plus - public_plus)**2
                      + np.linalg.norm(rotated_cross - public_cross)**2) / norm_b**2)

    seed = np.array([
        baseline["physical_fixed_distance_fit"]["lag_seconds"] * 1e3,
        baseline["physical_fixed_distance_fit"]["orbital_phase_rotation_rad"] * 2,
    ])
    bound = fd_plan["fit_lag_bound_seconds"] * 1e3
    optimum = minimize(
        residual, seed, method="Nelder-Mead",
        bounds=[(-bound, bound), (-math.pi, math.pi)],
        options={"initial_simplex": np.array([
            seed, seed + [0.05, 0.0], seed + [0.0, 0.01]]),
            "xatol": 1e-10, "fatol": 1e-15, "maxiter": 400},
    )
    if not optimum.success or optimum.fun > raw**2 + 1e-14:
        raise ValueError(f"physical fit failed: {optimum.message}")
    fit = float(math.sqrt(optimum.fun))
    return {
        "name": row["name"], "generation_seconds": generation_seconds,
        "native_fd_sha256": hashlib.sha256(
            native_arrays[0].tobytes() + native_arrays[1].tobytes()).hexdigest(),
        "public_fd_sha256": public_sha,
        "raw_complex_relative_l2": raw,
        "physical_fixed_distance_relative_l2": fit,
        "baseline_physical_fixed_distance_relative_l2": baseline[
            "physical_fixed_distance_fit"]["relative_l2"],
        "physical_fit_lag_seconds": float(optimum.x[0]) * 1e-3,
        "physical_fit_orbital_phase_rotation_rad": float(optimum.x[1]) / 2,
        "fit_lag_at_bound": bool(abs(optimum.x[0]) >= bound - 1e-7),
        "amplitude_relative_l2": float(np.linalg.norm(abs(a) - abs(b)) / norm_b),
        "native_to_public_norm_ratio": float(np.linalg.norm(a) / norm_b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-public-fd-intervention-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["safe_frequency_cap", "strong_antialigned"]
            or plan["baseline_report"] != "dev/native_rom/evidence/2026-09-25/native_public_fd.json"
            or plan["fit"] != "common lag and exact quadrupole polarization rotation at fixed distance and inclination; bounded Nelder-Mead, seeded from the frozen baseline fit"
            or plan["intervention"] != {
                "diagnostic_hamiltonian_log_derivative": "lal_v4",
                "diagnostic_hamiltonian_calibration": "lal_v4_rounded"}):
        parser.error("frozen intervention plan changed")
    baseline_path = q.ROOT / plan["baseline_report"]
    if q.sha256(baseline_path) != plan["baseline_report_sha256"]:
        parser.error("frozen baseline report differs")
    baseline = json.loads(baseline_path.read_text())
    frozen_binary = baseline["lal_binary"]
    if q.lal_binary_identity(frozen_binary["lal_version"],
                             frozen_binary["lalsimulation_version"]) != frozen_binary:
        parser.error("pinned public binary differs")
    fd_plan_path = q.ROOT / "dev/native_rom/evidence/2026-09-25/native_public_fd_plan.json"
    if q.sha256(fd_plan_path) != baseline["plan_sha256"]:
        parser.error("frozen baseline FD plan differs")
    fd_plan = json.loads(fd_plan_path.read_text())
    rows = {row["name"]: row for row in FROZEN_ROWS}
    previous = {row["name"]: row for row in baseline["rows"]}
    output = {
        "schema": "diffgw.seobnrv4.hamiltonian-public-fd-intervention.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_report_sha256": q.sha256(baseline_path),
        "process": {"host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
    }
    for name in plan["rows"]:
        try:
            result = _row(rows[name], previous[name], fd_plan)
            output["rows"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            output["errors"].append({"name": name, "error": repr(exc)})
            print(f"{name}: {exc!r}", flush=True)
    output["status"] = "measured" if not output["errors"] else "failed"
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if not output["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
