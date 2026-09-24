"""Post-hoc unweighted peak-window alignment for the five SXS cases.

This diagnoses fixed-frequency alignment offsets. It is not a
detector-noise-weighted faithfulness calculation or release gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import optimize_sxs_2123_alignment as fit
from . import qualify_generator as q


def write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False,
                               sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--public-control", type=Path, required=True)
    parser.add_argument("--alignment-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    primary = json.loads(args.primary.read_text())
    control = json.loads(args.public_control.read_text())
    alignment = json.loads(args.alignment_plan.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (alignment["schema"] != "diffgw.seobnrv4.sxs-five-case-alignment-plan.v1"
            or alignment["qualification_claim"] is not False
            or alignment["primary_sha256"] != q.sha256(args.primary)
            or alignment["public_control_sha256"] != q.sha256(args.public_control)
            or alignment["sxs_2139_pilot_sha256"] != q.sha256(
                args.alignment_plan.parent / "sxs_2139_alignment.json")
            or primary["plan_sha256"] != q.sha256(args.plan)
            or primary["extraction_sha256"] != q.sha256(extraction_path)
            or control["plan_sha256"] != q.sha256(args.plan)
            or not primary["all_cases_succeeded"]
            or not control["all_cases_succeeded"]
            or set(extraction["cases"]) != set(plan["cases"])):
        raise ValueError("alignment prerequisites changed")
    torch.set_num_threads(1)
    method = plan["method"]
    dt = method["grid_step_M"]
    window = tuple(alignment["window_relative_to_reference_peak_M"])
    bounds = tuple(alignment["allowed_time_shift_M"])
    source_before = q.canonical_hash(q.source_snapshot())
    result = dict(schema="diffgw.seobnrv4.sxs-five-case-alignment.v1",
                  plan_sha256=q.sha256(args.plan),
                  primary_sha256=q.sha256(args.primary),
                  public_control_sha256=q.sha256(args.public_control),
                  alignment_plan_sha256=q.sha256(args.alignment_plan),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=source_before,
                  detector_noise_weighted=False, release_qualified=False,
                  status="running", cases={})
    write(args.json_out, result)
    for simulation, case in plan["cases"].items():
        started = time.monotonic()
        entry = dict(errors=[])
        result["cases"][simulation] = entry
        try:
            native = q.native_run(case["row"], dict(name="native",
                atol=method["native_atol"], rtol=method["native_rtol"], dt_geom=dt),
                q.Settings(max_run_seconds=600.),
                diagnostic_fine_rkf45_resampling=
                    method["diagnostic_fine_rkf45_resampling"])
            if (q.summarize_run(native)["array_sha256"]
                    != primary["cases"][simulation]["native"]["array_sha256"]):
                raise ValueError("native waveform changed")
            high, low = max(case["levels"]), min(case["levels"])
            nr = {}
            for lev in (low, high):
                record = extraction["cases"][simulation]["extracted"][f"Lev{lev}"]
                path = args.decoded_dir / record["path"]
                if q.sha256(path) != record["sha256"]:
                    raise ValueError(f"NR Lev{lev} mode changed")
                nr[lev] = c.anchor(c.load_mode(path), dt=dt)
            waves = dict(native=c.anchor(native, dt=dt),
                         public=c.anchor(public.public_mode(
                             case["row"], dt, alignment["public_call"]), dt=dt),
                         low=nr[low], high=nr[high])
            entry["pairs"] = {
                name: fit.optimized_pair(waves[candidate], waves[reference],
                                         window=window, dt=dt, lag_bounds=bounds)
                for name, candidate, reference in (
                    ("native_vs_public", "native", "public"),
                    ("native_vs_high", "native", "high"),
                    ("public_vs_high", "public", "high"),
                    ("low_vs_high", "low", "high"))}
            entry["elapsed_seconds"] = time.monotonic()-started
            print(f"{simulation} complete", flush=True)
        except Exception as exc:
            entry["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
            entry["elapsed_seconds"] = time.monotonic()-started
            print(f"{simulation} failed: {exc}", flush=True)
        write(args.json_out, result)
    result["generator_source_manifest_sha256_after"] = q.canonical_hash(q.source_snapshot())
    result["sources_unchanged"] = (result["generator_source_manifest_sha256_after"]
                                   == source_before)
    result["all_cases_succeeded"] = all("pairs" in row and not row["errors"]
                                        for row in result["cases"].values())
    result["status"] = "diagnostic_complete" if result["sources_unchanged"] else "source_changed"
    write(args.json_out, result)
    return 0 if result["sources_unchanged"] and result["all_cases_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
