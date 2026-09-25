"""Prospective development check of a fixed wider ROM-input phase gauge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from . import qualify_generator as q
from .diagnose_rom_input_stability import _run, _support
from .rom.conditioning import condition_modes_windowed


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_input_windowed_gauge_plan.json"


def _compare(first: np.ndarray, second: np.ndarray, frequency: np.ndarray,
             reference_frequency: float, width: float) -> dict[str, float]:
    a = condition_modes_windowed(frequency, first, reference_frequency,
                                 half_width_hz=width)
    b = condition_modes_windowed(frequency, second, reference_frequency,
                                 half_width_hz=width)
    ca = np.exp(a.log_amplitude + 1j * a.phase)
    cb = np.exp(b.log_amplitude + 1j * b.phase)
    norm = np.linalg.norm(cb)
    return {
        "complex_relative_l2": float(np.linalg.norm(ca - cb) / norm),
        "amplitude_relative_l2": float(np.linalg.norm(abs(first) - abs(second)) / norm),
        "phase_rms_rad": float(np.sqrt(np.mean((a.phase - b.phase) ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--spectra-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.rom-input-windowed-gauge-plan.v1"
            or plan["release_qualified"] or len(plan["f_max_levels_hz"]) != 3
            or plan["gauge_half_width_hz"] != 2.0):
        parser.error("unexpected frozen development plan")
    low, high = plan["comparison_band_hz"]
    df = plan["delta_f_hz"]
    frequency = np.arange(round(low / df), round(high / df) + 1) * df
    if frequency[0] != low or frequency[-1] != high:
        parser.error("comparison band is not on the frozen grid")
    report = {
        "schema": "diffgw.seobnrv4.rom-input-windowed-gauge.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "conditioner_sha256": q.sha256(q.ROOT / "dev/native_rom/rom/conditioning.py"),
        "source_snapshot_sha256": q.canonical_hash(q.source_snapshot()),
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
    }
    spectra = {"frequency_hz": frequency}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.spectra_out.parent.mkdir(parents=True, exist_ok=True)
    for row in plan["diagnostic_rows"]:
        item = {"name": row["name"], "levels": [], "pairs": [], "passed": False}
        try:
            measured = []
            for index, fmax in enumerate(plan["f_max_levels_hz"]):
                level = {"name": f"grid_{index}", "atol": plan["atol"],
                         "rtol": plan["rtol"], "f_max_hz": fmax}
                run = _run(row, level, plan, frequency)
                measured.append(run)
                summary = {key: value for key, value in run.items()
                           if key not in ("plus", "cross")}
                for polarization in ("plus", "cross"):
                    signal = run[polarization]
                    spectra[f"{row['name']}_grid_{index}_{polarization}"] = signal
                    summary[f"{polarization}_sha256"] = hashlib.sha256(signal.tobytes()).hexdigest()
                    summary[f"{polarization}_support"] = _support(signal, plan["gates"])
                item["levels"].append(summary)
                np.savez(args.spectra_out, **spectra)
                args.json_out.write_text(json.dumps(report | {"rows": [*report["rows"], item]},
                                                    indent=2, sort_keys=True) + "\n")
                print(f"{row['name']} grid_{index}: {run['elapsed_seconds']:.1f}s", flush=True)
            for first, second in zip(measured, measured[1:]):
                pair = {"from": first["level"], "to": second["level"]}
                for polarization in ("plus", "cross"):
                    result = _compare(first[polarization], second[polarization], frequency,
                                      plan["reference_frequency_hz"],
                                      plan["gauge_half_width_hz"])
                    pair[polarization] = result
                    pair[f"{polarization}_passed"] = (
                        result["complex_relative_l2"] <= plan["gates"]["maximum_adjacent_complex_relative_l2"]
                        and result["amplitude_relative_l2"] <= plan["gates"]["maximum_adjacent_amplitude_relative_l2"]
                        and result["phase_rms_rad"] <= plan["gates"]["maximum_adjacent_phase_rms_rad"]
                    )
                item["pairs"].append(pair)
            item["passed"] = (all(pair[f"{pol}_passed"] for pair in item["pairs"]
                                  for pol in ("plus", "cross"))
                              and all(all(level[f"{pol}_support"]["checks"].values())
                                      for level in item["levels"] for pol in ("plus", "cross")))
        except Exception as exc:
            report["errors"].append({"name": row["name"], "error": repr(exc)})
            print(f"{row['name']}: {exc!r}", flush=True)
        report["rows"].append(item)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.spectra_out, **spectra)
    report["spectra_sha256"] = q.sha256(args.spectra_out)
    report["status"] = "measured" if not report["errors"] else "failed"
    report["all_checks_passed"] = not report["errors"] and all(row["passed"] for row in report["rows"])
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
