"""Frozen, development-only high-cadence continuation of the q4 FD input check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from . import qualify_generator as q
from .diagnose_rom_input_shared_epoch import _pair, _run, _shared_gauge, _synthetic_control
from .diagnose_rom_input_stability import _support


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/rom_q4_high_cadence_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--spectra-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    prior_dir = q.ROOT / "dev/native_rom/evidence/2026-09-25"
    prior_report = prior_dir / "rom_input_stability.json"
    prior_spectra = prior_dir / "rom_input_stability_spectra.npz"
    shared_runner = q.ROOT / "dev/native_rom/diagnose_rom_input_shared_epoch.py"
    conditioner = q.ROOT / "dev/native_rom/rom/conditioning.py"
    adapter = q.ROOT / "dev/native_rom/eob/torch/conditioned_waveform_clean.py"
    td_fd = q.ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py"
    if (plan["schema"] != "diffgw.seobnrv4.rom-q4-high-cadence-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row"]["name"] != "q4_mixed_spin"
            or plan["f_max_levels_hz"] != [16384.0, 32768.0, 65536.0]
            or plan["reserved_holdouts_remain_unevaluated"]
            != ["off_grid_mixed_spin", "high_q_antialigned"]
            or q.sha256(__file__) != plan["runner_sha256"]
            or q.sha256(shared_runner) != plan["shared_runner_sha256"]
            or q.sha256(conditioner) != plan["conditioner_sha256"]
            or q.sha256(adapter) != plan["fd_adapter_sha256"]
            or q.sha256(td_fd) != plan["td_fd_conditioner_sha256"]
            or q.sha256(prior_report) != plan["prior_report_sha256"]
            or q.sha256(prior_spectra) != plan["prior_spectra_sha256"]
            or q.canonical_hash(q.source_snapshot()) != plan["source_snapshot_sha256"]):
        parser.error("frozen plan, source, prior spectra, or code changed")
    prior = json.loads(prior_report.read_text())
    prior_row = next(row for row in prior["rows"] if row["name"] == "q4_mixed_spin")
    previous = next(level for level in prior_row["levels"] if level["level"] == "ode_2")
    df = plan["delta_f_hz"]
    low, high = plan["comparison_band_hz"]
    frequency = np.arange(round(low / df), round(high / df) + 1) * df
    if (frequency[0] != low or frequency[-1] != high
            or plan["reference_frequency_hz"] != 100.0
            or plan["gauge_half_width_hz"] != 2.0):
        parser.error("invalid frozen frequency/gauge grid")
    synthetic = _synthetic_control(plan)
    report = {
        "schema": "diffgw.seobnrv4.rom-q4-high-cadence.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "runner_sha256": q.sha256(__file__),
        "source_snapshot_sha256": plan["source_snapshot_sha256"],
        "prior_report_sha256": q.sha256(prior_report),
        "prior_spectra_sha256": q.sha256(prior_spectra),
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "synthetic_control": synthetic,
        "row": plan["row"], "levels": [], "pairs": [], "checks": {}, "errors": [],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.spectra_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    spectra = {"frequency_hz": frequency}
    if not synthetic["passed"]:
        report["errors"].append("synthetic convention control failed")
    else:
        try:
            runs = []
            gauges = []
            for index, level in enumerate(plan["f_max_levels_hz"]):
                run = _run(plan["row"], level, plan, frequency)
                gauge = _shared_gauge(run, frequency, plan)
                runs.append(run)
                gauges.append(gauge)
                summary = {key: value for key, value in run.items()
                           if not isinstance(value, np.ndarray)}
                summary["shared_gauge_removed_plus_phase_rad"] = gauge["removed_plus_phase_rad"]
                summary["shared_gauge_removed_plus_slope_rad_per_hz"] = gauge[
                    "removed_plus_slope_rad_per_hz"]
                for pol in ("plus", "cross"):
                    for name in (pol, f"{pol}_pad_crop_only", f"{pol}_physical_epoch"):
                        value = run[name]
                        spectra[f"grid_{index}_{name}"] = value
                        summary[f"{name}_sha256"] = hashlib.sha256(value.tobytes()).hexdigest()
                    summary[f"{pol}_support"] = _support(
                        run[f"{pol}_physical_epoch"], plan["gates"])
                report["levels"].append(summary)
                np.savez(args.spectra_out, **spectra)
                args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                print(f"grid_{index} {level:g} Hz: {run['elapsed_seconds']:.1f}s", flush=True)
            report["pairs"] = [_pair(a, b, ga, gb, frequency, plan)
                               for a, b, ga, gb in zip(runs, runs[1:], gauges, gauges[1:])]
            coarse, fine = report["pairs"]
            threshold = plan["gates"]["fine_to_coarse_shared_gauge_complex_error_ratio"]
            report["checks"] = {
                "replay_original_fine_raw": all(
                    report["levels"][0][f"{pol}_sha256"] == previous[f"{pol}_sha256"]
                    for pol in ("plus", "cross")
                ),
                "pair_budgets": all(all(pair["checks"].values()) for pair in report["pairs"]),
                "support_and_unwrap": all(
                    all(level[f"{pol}_support"]["checks"].values())
                    for level in report["levels"] for pol in ("plus", "cross")
                ),
                "nonincreasing_fine_complex_error": all(
                    fine["shared_gauge"][pol]["complex_relative_l2"]
                    <= threshold * coarse["shared_gauge"][pol]["complex_relative_l2"] + 1e-12
                    for pol in ("plus", "cross")
                ),
            }
        except Exception as exc:
            report["errors"].append(repr(exc))
    np.savez(args.spectra_out, **spectra)
    report["spectra_sha256"] = q.sha256(args.spectra_out)
    report["all_checks_passed"] = (synthetic["passed"] and not report["errors"]
                                   and all(report["checks"].values()))
    report["status"] = "measured" if not report["errors"] else "failed"
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
