"""Development-only prospective FD input stability pilot; never fit ROM data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import time

import numpy as np
import torch

from . import qualify_generator as q
from .eob.torch.conditioned_waveform_clean import conditioned_seobnrv4_fd
from .rom.conditioning import condition_modes


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_stability_plan.json"


@contextmanager
def deadline(seconds: float):
    def expired(*_):
        raise TimeoutError(f"FD generation exceeded {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run(row: dict, level: dict, plan: dict, band: np.ndarray) -> dict:
    def generator(*args, **kwargs):
        from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive

        return reconstruct_seobnrv4_22_adaptive(
            *args, **kwargs, atol=level["atol"], rtol=level["rtol"]
        )

    start = time.monotonic()
    with deadline(plan["per_run_timeout_seconds"]), torch.no_grad():
        result = conditioned_seobnrv4_fd(
            row["mass1"], row["mass2"], chi1z=row["chi1"], chi2z=row["chi2"],
            f_low=row["f_low"], f_max=level["f_max_hz"],
            delta_f=plan["delta_f_hz"], distance=plan["distance_mpc"],
            inclination=plan["inclination_rad"],
            ringdown_duration=plan["ringdown_duration_M"],
            generator=generator, return_metadata=True,
        )
    frequency = result.frequencies.detach().cpu().numpy()
    indices = np.searchsorted(frequency, band)
    if not np.array_equal(frequency[indices], band):
        raise ValueError("FD frequency grid does not contain the frozen band")
    plus = result.h_plus.detach().cpu().numpy()[indices]
    cross = result.h_cross.detach().cpu().numpy()[indices]
    if not np.isfinite(plus).all() or not np.isfinite(cross).all():
        raise ValueError("nonfinite FD spectrum")
    return {
        "level": level["name"], "elapsed_seconds": time.monotonic() - start,
        "td_samples": int(result.td_time.numel()),
        "fft_samples": int(result.sampling.n_time),
        "dt_geom": float(result.dt_geom),
        "omega_start": float(result.omega_start),
        "plus": plus, "cross": cross,
    }


def _support(signal: np.ndarray, gates: dict) -> dict:
    amplitude = abs(signal)
    maximum = float(np.max(amplitude))
    minimum_relative = float(np.min(amplitude) / maximum) if maximum else 0.0
    phase = np.unwrap(np.angle(signal))
    decimated = np.unwrap(np.angle(signal[::2]))
    difference = phase[::2] - decimated
    difference -= 2 * np.pi * np.rint(difference[0] / (2 * np.pi))
    max_step = float(np.max(abs(np.diff(phase))))
    max_decimated = float(np.max(abs(difference)))
    checks = {
        "amplitude_support": minimum_relative >= gates["minimum_relative_amplitude_in_band"],
        "adjacent_phase_step": max_step <= gates["maximum_unwrapped_adjacent_phase_step_rad"],
        "decimated_unwrap": max_decimated <= gates["maximum_decimated_phase_unwrap_difference_rad"],
    }
    return {
        "minimum_relative_amplitude": minimum_relative,
        "maximum_adjacent_phase_step_rad": max_step,
        "maximum_decimated_unwrap_difference_rad": max_decimated,
        "checks": checks,
    }


def _comparison(candidate: np.ndarray, reference: np.ndarray,
                frequency: np.ndarray, f_ref: float) -> dict:
    a = condition_modes(frequency, candidate, f_ref)
    b = condition_modes(frequency, reference, f_ref)
    ca = np.exp(a.log_amplitude + 1j * a.phase)
    cb = np.exp(b.log_amplitude + 1j * b.phase)
    norm = np.linalg.norm(cb)
    phase_difference = a.phase - b.phase
    return {
        "complex_relative_l2": float(np.linalg.norm(ca - cb) / norm),
        "amplitude_relative_l2": float(np.linalg.norm(abs(candidate) - abs(reference)) / norm),
        "phase_rms_rad": float(np.sqrt(np.mean(phase_difference**2))),
        "phase_max_abs_rad": float(np.max(abs(phase_difference))),
    }


def _convergence(values: list[dict], frequency: np.ndarray, plan: dict) -> dict:
    first = _comparison(values[0]["plus"], values[1]["plus"], frequency,
                        plan["reference_frequency_hz"])
    last = _comparison(values[1]["plus"], values[2]["plus"], frequency,
                       plan["reference_frequency_hz"])
    gates = plan["gates"]
    checks = {
        "complex_budget": last["complex_relative_l2"] <= gates["medium_to_fine_complex_relative_l2"],
        "amplitude_budget": last["amplitude_relative_l2"] <= gates["medium_to_fine_amplitude_relative_l2"],
        "phase_budget": last["phase_rms_rad"] <= gates["medium_to_fine_phase_rms_rad"],
        "complex_refinement": last["complex_relative_l2"] <= first["complex_relative_l2"] + 1e-12,
    }
    return {"coarse_to_medium": first, "medium_to_fine": last,
            "checks": checks, "passed": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--spectra-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if plan["schema"] != "diffgw.seobnrv4.rom-input-stability-plan.v1" or plan["release_qualified"]:
        parser.error("unexpected pilot plan")
    if (len(plan["f_max_levels_hz"]) != 3 or len(plan["atol_levels"]) != 3
            or len(plan["rtol_levels"]) != 3
            or any(not math.isfinite(v) or v <= 0 for v in
                   (*plan["f_max_levels_hz"], *plan["atol_levels"], *plan["rtol_levels"]))):
        parser.error("invalid convergence levels")
    low, high = plan["comparison_band_hz"]
    frequency = np.arange(round(low / plan["delta_f_hz"]),
                          round(high / plan["delta_f_hz"]) + 1) * plan["delta_f_hz"]
    if (frequency[0] != low or frequency[-1] != high
            or not low < plan["reference_frequency_hz"] < high
            or high >= min(plan["f_max_levels_hz"])):
        parser.error("invalid frozen frequency band")
    tight = dict(atol=plan["atol_levels"][-1], rtol=plan["rtol_levels"][-1])
    levels = [dict(name=f"ode_{i}", atol=atol, rtol=rtol,
                   f_max_hz=plan["f_max_levels_hz"][-1])
              for i, (atol, rtol) in enumerate(zip(plan["atol_levels"], plan["rtol_levels"]))]
    levels += [dict(name=f"grid_{i}", **tight, f_max_hz=fmax)
               for i, fmax in enumerate(plan["f_max_levels_hz"][:-1])]
    report = {
        "schema": "diffgw.seobnrv4.rom-input-stability.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "source_snapshot_sha256": q.canonical_hash(q.source_snapshot()),
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__,
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "frequency_samples": int(len(frequency)),
        "rows": [], "errors": [],
        "scope": "Selected native FD numerical stability only; no ROM fit, independent source fidelity, or NR validation.",
    }
    spectra = {"frequency_hz": frequency}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.spectra_out.parent.mkdir(parents=True, exist_ok=True)
    for row in plan["diagnostic_rows"]:
        measured = []
        row_report = {"name": row["name"], "levels": [], "checks": {}}
        try:
            for level in levels:
                item = _run(row, level, plan, frequency)
                measured.append(item)
                name = f"{row['name']}_{level['name']}"
                spectra[name + "_plus"] = item["plus"]
                spectra[name + "_cross"] = item["cross"]
                summary = {key: value for key, value in item.items()
                           if key not in ("plus", "cross")}
                summary["plus_sha256"] = hashlib.sha256(item["plus"].tobytes()).hexdigest()
                summary["cross_sha256"] = hashlib.sha256(item["cross"].tobytes()).hexdigest()
                summary["plus_support"] = _support(item["plus"], plan["gates"])
                row_report["levels"].append(summary)
                np.savez(args.spectra_out, **spectra)
                checkpoint = dict(report, rows=[*report["rows"], row_report])
                args.json_out.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
                print(f"{name}: {item['elapsed_seconds']:.1f}s", flush=True)
            grid = [measured[3], measured[4], measured[2]]
            row_report["ode_convergence"] = _convergence(measured[:3], frequency, plan)
            row_report["grid_convergence"] = _convergence(grid, frequency, plan)
            row_report["checks"] = {
                "fd_support": all(all(level["plus_support"]["checks"].values())
                                  for level in row_report["levels"]),
                "ode_convergence": row_report["ode_convergence"]["passed"],
                "grid_convergence": row_report["grid_convergence"]["passed"],
            }
        except Exception as exc:
            report["errors"].append({"name": row["name"], "error": repr(exc)})
            print(f"{row['name']}: {exc!r}", flush=True)
        report["rows"].append(row_report)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.spectra_out, **spectra)
    report["spectra_sha256"] = q.sha256(args.spectra_out)
    report["status"] = ("measured" if not report["errors"] else "failed")
    report["all_numerical_checks_passed"] = (
        not report["errors"] and all(all(row["checks"].values()) for row in report["rows"])
    )
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
