"""Compare the frozen third SXS holdout with the source-backed native h22.

This is a development-only selected-case proxy, not release qualification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from . import compare_sxs_2123 as c
from . import qualify_generator as q

SCHEMAS = {
    "diffgw.seobnrv4.sxs-third-holdout-plan.v1": (
        "diffgw.seobnrv4.sxs-third-holdout-extraction.v1",
        "diffgw.seobnrv4.sxs-third-holdout-comparison.v1"),
}


def write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--array-dir", type=Path,
                        help="Optional directory for anchored native and NR arrays")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan.get("schema") not in SCHEMAS or plan.get("qualification_claim") is not False
            or len(plan.get("cases", {})) != 5):
        raise ValueError("unexpected five-case comparison plan")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (extraction["schema"] != SCHEMAS[plan["schema"]][0]
            or extraction["plan_sha256"] != q.sha256(args.plan)
            or set(extraction["cases"]) != set(plan["cases"])):
        raise ValueError("decoded NR does not match frozen plan")
    choice_path = q.ROOT / plan["candidate"]["source_choice_path"]
    if q.sha256(choice_path) != plan["candidate"]["source_choice_sha256"]:
        raise ValueError("source-backed candidate choice changed")
    choices = json.loads(choice_path.read_text())["choice"]
    numerical = plan["candidate"]["numerical"]
    if (numerical["atol"] != plan["method"]["native_atol"]
            or numerical["rtol"] != plan["method"]["native_rtol"]
            or numerical["nqc_phase_window_M"] != choices["nqc_phase_window_M"]
            or numerical["restart_time_M"] is not None):
        raise ValueError("candidate numerical choices changed")
    torch.set_num_threads(1)
    source = q.canonical_hash(q.source_snapshot())
    result = dict(schema=SCHEMAS[plan["schema"]][1],
                  scope=q.SCOPE, status="running", physical_validation_passed=False,
                  release_qualified=False, plan_sha256=q.sha256(args.plan),
                  extraction_sha256=q.sha256(extraction_path),
                  script_sha256=q.sha256(__file__),
                  source_choice_sha256=q.sha256(choice_path),
                  generator_source_manifest_sha256=source,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   text=True).strip(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv), cases={})
    write(args.json_out, result)
    method = plan["method"]
    dt = method["grid_step_M"]
    windows = method["windows_relative_to_highest_resolution_amplitude_peak_M"]
    for simulation, case in plan["cases"].items():
        started = time.monotonic()
        entry = dict(row=case["row"], levels=case["levels"], errors=[],
                     physical_validation_passed=False, release_qualified=False)
        result["cases"][simulation] = entry
        try:
            decoded = extraction["cases"][simulation]
            if set(decoded["archive_files"]) != set(case["files"]):
                raise ValueError("archive file inventory changed")
            for name, details in case["files"].items():
                if decoded["archive_files"][name]["md5"] != details["md5"]:
                    raise ValueError(f"archive checksum mismatch: {name}")
            nr = {}
            for lev in case["levels"]:
                record = decoded["extracted"][f"Lev{lev}"]
                path = args.decoded_dir / record["path"]
                if q.sha256(path) != record["sha256"]:
                    raise ValueError(f"decoded mode checksum mismatch: Lev{lev}")
                nr[lev] = c.anchor(c.load_mode(path), dt=dt)
            native = q.native_run(case["row"], dict(name="source_candidate",
                atol=method["native_atol"], rtol=method["native_rtol"],
                dt_geom=dt), q.Settings(max_run_seconds=method.get(
                    "max_run_seconds_per_case", 600.)),
                diagnostic_fine_rkf45_resampling=numerical["fine_rkf45_resampling"],
                diagnostic_matching_second_derivatives=
                    numerical["matching_second_derivatives"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_derivatives"],
                diagnostic_nqc_phase_window_M=numerical["nqc_phase_window_M"],
                diagnostic_restart_time_M=numerical["restart_time_M"],
                diagnostic_coarse_initial_step_M=numerical["coarse_initial_step_M"],
                diagnostic_spin_truncation=choices["spin_truncation"],
                diagnostic_mode_21_higher_spin=choices["flux"]["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=choices["flux"]["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=choices["flux"]["mode_32_spin_square"],
                diagnostic_mode_33_higher_spin=choices["physical"]["mode_33_higher_spin"],
                diagnostic_mode_43_higher_spin=choices["mode_43_higher_spin"],
                diagnostic_hamiltonian_log_derivative=
                    choices["physical"]["hamiltonian_log_derivative"],
                diagnostic_hamiltonian_calibration=choices["calibration"])
            anchored = c.anchor(native, dt=dt)
            low, high = case["levels"]
            if args.array_dir is not None:
                args.array_dir.mkdir(parents=True, exist_ok=True)
                array_path = args.array_dir / f"{simulation.replace(':', '_')}_anchored.npz"
                np.savez(array_path, **{
                    f"{name}_{key}": wave[key]
                    for name, wave in (("native", anchored), ("nr_low", nr[low]),
                                       ("nr_high", nr[high]))
                    for key in ("time", "mode")})
                entry["anchored_array"] = dict(path=str(array_path),
                                               sha256=q.sha256(array_path))
            support = (float(anchored["time"][0]), float(anchored["time"][-1]))
            entry.update(native=q.summarize_run(native), anchor={
                name: {key: value for key, value in waveform.items()
                       if key not in ("time", "mode")}
                for name, waveform in (("native", anchored),
                                       (f"Lev{low}", nr[low]), (f"Lev{high}", nr[high]))},
                comparisons=dict(
                    native_vs_high=c.compare(anchored, nr[high], windows, dt),
                    low_vs_high_on_native_support=c.compare(
                        nr[low], nr[high], windows, dt, support=support)),
                elapsed_seconds=time.monotonic()-started)
            print(f"{simulation} complete", flush=True)
        except Exception as exc:
            entry["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
            entry["elapsed_seconds"] = time.monotonic()-started
            print(f"{simulation} failed: {exc}", flush=True)
        write(args.json_out, result)
    result["generator_source_manifest_sha256_after"] = q.canonical_hash(q.source_snapshot())
    result["source_choice_sha256_after"] = q.sha256(choice_path)
    result["sources_unchanged"] = (
        result["generator_source_manifest_sha256_after"] == source
        and result["source_choice_sha256_after"] == result["source_choice_sha256"])
    result["all_cases_succeeded"] = all(
        "comparisons" in entry and not entry["errors"]
        for entry in result["cases"].values())
    result["status"] = "diagnostic_complete" if result["sources_unchanged"] else "source_changed"
    write(args.json_out, result)
    print(json.dumps(dict(status=result["status"],
                          all_cases_succeeded=result["all_cases_succeeded"])), flush=True)
    return 0 if result["sources_unchanged"] and result["all_cases_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
