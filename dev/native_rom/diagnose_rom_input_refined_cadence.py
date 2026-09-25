"""Development-only fourth-cadence continuation of the failed FD-input pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from . import qualify_generator as q
from .diagnose_rom_input_stability import _comparison, _run, _support


BASE = q.ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = BASE / "rom_input_refined_cadence_plan.json"
OLD_PLAN = BASE / "rom_input_stability_plan.json"
OLD_REPORT = BASE / "rom_input_stability.json"
OLD_SPECTRA = BASE / "rom_input_stability_spectra.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--spectra-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    old_plan = json.loads(OLD_PLAN.read_text())
    old = json.loads(OLD_REPORT.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.rom-input-refined-cadence-plan.v1"
            or plan["release_qualified"]
            or plan["prior_report_sha256"] != q.sha256(OLD_REPORT)
            or plan["prior_spectra_sha256"] != q.sha256(OLD_SPECTRA)
            or old["source_snapshot_sha256"] != q.canonical_hash(q.source_snapshot())
            or plan["f_max_hz"] != 32768.0):
        parser.error("prior evidence, source, or frozen refined plan differs")
    low, high = old_plan["comparison_band_hz"]
    frequency = np.arange(round(low / old_plan["delta_f_hz"]),
                          round(high / old_plan["delta_f_hz"]) + 1) * old_plan["delta_f_hz"]
    level = dict(name="grid_3", atol=old_plan["atol_levels"][-1],
                 rtol=old_plan["rtol_levels"][-1], f_max_hz=plan["f_max_hz"])
    settings = dict(old_plan, per_run_timeout_seconds=plan["per_run_timeout_seconds"])
    report = {
        "schema": "diffgw.seobnrv4.rom-input-refined-cadence.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "prior_report_sha256": q.sha256(OLD_REPORT),
        "prior_spectra_sha256": q.sha256(OLD_SPECTRA),
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
        "scope": "Fourth-cadence numerical diagnostic on two reused rows; prior failed gate remains failed.",
    }
    spectra = {"frequency_hz": frequency}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.spectra_out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(OLD_SPECTRA) as old_spectra:
        if not np.array_equal(old_spectra["frequency_hz"], frequency):
            parser.error("prior frequency band differs")
        for row, prior in zip(old_plan["diagnostic_rows"], old["rows"], strict=True):
            if row["name"] != prior["name"]:
                parser.error("prior row order differs")
            try:
                item = _run(row, level, settings, frequency)
                old_plus = old_spectra[f"{row['name']}_ode_2_plus"]
                if hashlib.sha256(old_plus.tobytes()).hexdigest() != prior["levels"][2]["plus_sha256"]:
                    parser.error("prior fine spectrum differs")
                metrics = _comparison(old_plus, item["plus"], frequency,
                                      old_plan["reference_frequency_hz"])
                support = _support(item["plus"], old_plan["gates"])
                prior_metrics = prior["grid_convergence"]["medium_to_fine"]
                gates = plan["checks"]
                checks = {
                    "complex_refinement": metrics["complex_relative_l2"] <=
                    gates["new_complex_relative_l2_at_most_previous_times"]
                    * prior_metrics["complex_relative_l2"],
                    "phase_refinement": metrics["phase_rms_rad"] <=
                    gates["new_phase_rms_at_most_previous_times"]
                    * prior_metrics["phase_rms_rad"],
                    "amplitude_budget": metrics["amplitude_relative_l2"] <=
                    gates["new_amplitude_relative_l2_at_most"],
                    "frequency_support_and_unwrap": all(support["checks"].values()),
                }
                spectra[row["name"] + "_grid_3_plus"] = item["plus"]
                spectra[row["name"] + "_grid_3_cross"] = item["cross"]
                report["rows"].append({
                    "name": row["name"], "elapsed_seconds": item["elapsed_seconds"],
                    "td_samples": item["td_samples"], "fft_samples": item["fft_samples"],
                    "dt_geom": item["dt_geom"],
                    "plus_sha256": hashlib.sha256(item["plus"].tobytes()).hexdigest(),
                    "cross_sha256": hashlib.sha256(item["cross"].tobytes()).hexdigest(),
                    "prior_pair": prior_metrics, "new_pair": metrics,
                    "plus_support": support, "checks": checks,
                    "passed": all(checks.values()),
                })
                np.savez(args.spectra_out, **spectra)
                args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                print(f"{row['name']}: {item['elapsed_seconds']:.1f}s {checks}", flush=True)
            except Exception as exc:
                report["errors"].append({"name": row["name"], "error": repr(exc)})
                print(f"{row['name']}: {exc!r}", flush=True)
    np.savez(args.spectra_out, **spectra)
    report["spectra_sha256"] = q.sha256(args.spectra_out)
    report["status"] = "measured" if not report["errors"] else "failed"
    report["all_checks_passed"] = not report["errors"] and all(
        row["passed"] for row in report["rows"]
    )
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
