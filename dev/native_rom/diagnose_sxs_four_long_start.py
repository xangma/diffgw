"""Frozen four-case earlier-start SXS diagnostic; not release validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import optimize_sxs_2123_alignment as fit
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--public-control", type=Path, required=True)
    parser.add_argument("--long-plan", type=Path, required=True)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.case_plan.read_text())
    long_plan = json.loads(args.long_plan.read_text())
    primary = json.loads(args.primary.read_text())
    alignment = json.loads(args.alignment.read_text())
    control = json.loads(args.public_control.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    expected = {
        "case_plan_sha256": args.case_plan,
        "primary_sha256": args.primary,
        "alignment_sha256": args.alignment,
        "public_control_sha256": args.public_control,
        "extraction_sha256": extraction_path,
    }
    if (long_plan["schema"] != "diffgw.seobnrv4.sxs-four-long-start-plan.v1"
            or long_plan["qualification_claim"] is not False
            or any(long_plan[key] != q.sha256(path) for key, path in expected.items())
            or args.simulation not in long_plan["omega_starts"]
            or not primary["all_cases_succeeded"]
            or not alignment["all_cases_succeeded"]
            or not control["all_cases_succeeded"]
            or primary["plan_sha256"] != q.sha256(args.case_plan)
            or primary["extraction_sha256"] != q.sha256(extraction_path)):
        raise ValueError("four-case long-start prerequisites changed")
    torch.set_num_threads(1)
    simulation = args.simulation
    case = plan["cases"][simulation]
    method = plan["method"]
    dt = method["grid_step_M"]
    if dt != long_plan["grid_step_M"]:
        raise ValueError("sample step changed")
    high = max(case["levels"])
    metadata_path = (args.archive_dir / simulation.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    metadata = json.loads(metadata_path.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    omega_start = long_plan["omega_starts"][simulation]
    if (not 0 < omega_start < orbital or
            not math.isclose(metadata["reference_time"],
                             metadata["relaxation_time"], abs_tol=1e-12)):
        raise ValueError("early start is not before settled NR reference")
    row = dict(case["row"], omega_start=omega_start)
    record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / record["path"]
    if q.sha256(nr_path) != record["sha256"]:
        raise ValueError("NR mode changed")
    source_before = q.canonical_hash(q.source_snapshot())
    started = time.monotonic()
    native = q.native_run(
        row, dict(name="native_long_start", atol=method["native_atol"],
                  rtol=method["native_rtol"], dt_geom=dt),
        q.Settings(max_run_seconds=long_plan["max_run_seconds_per_case"]),
        diagnostic_fine_rkf45_resampling=method["diagnostic_fine_rkf45_resampling"])
    public_wave = public.public_mode(row, dt, long_plan["public_call"])
    waves = dict(native=c.anchor(native, dt=dt),
                 public=c.anchor(public_wave, dt=dt),
                 nr=c.anchor(c.load_mode(nr_path), dt=dt))
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, **{
        f"{name}_{key}": wave[key] for name, wave in waves.items()
        for key in ("time", "mode")})
    window = tuple(long_plan["window_relative_to_reference_peak_M"])
    bounds = tuple(long_plan["allowed_time_shift_M"])
    pairs = {
        name: fit.optimized_pair(waves[candidate], waves[reference],
                                 window=window, dt=dt, lag_bounds=bounds)
        for name, candidate, reference in (
            ("native_vs_nr", "native", "nr"),
            ("public_vs_nr", "public", "nr"),
            ("native_vs_public", "native", "public"))}
    source_after = q.canonical_hash(q.source_snapshot())
    result = dict(schema="diffgw.seobnrv4.sxs-four-long-start.v1",
                  simulation=simulation, plan_sha256=q.sha256(args.case_plan),
                  long_plan_sha256=q.sha256(args.long_plan),
                  script_sha256=q.sha256(__file__),
                  source_manifest_sha256_before=source_before,
                  source_manifest_sha256_after=source_after,
                  sources_unchanged=source_before == source_after,
                  nr_high_metadata_sha256=q.sha256(metadata_path),
                  nr_relaxation_time_M=metadata["relaxation_time"],
                  nr_reference_orbital_MOmega=orbital,
                  row=row, native=q.summarize_run(native),
                  public_call={key: public_wave[key] for key in (
                      "f_lower_Hz", "delta_t_s", "distance_Mpc", "inclination_rad")},
                  anchors={name: {key: value for key, value in wave.items()
                                  if key not in ("time", "mode")}
                           for name, wave in waves.items()},
                  array_sha256=q.sha256(args.array_out),
                  direct_native_vs_nr=c.compare(
                      waves["native"], waves["nr"],
                      method["windows_relative_to_highest_resolution_amplitude_peak_M"], dt),
                  pairs=pairs,
                  late_start_native_vs_nr_overlap=alignment["cases"][simulation]
                      ["pairs"]["native_vs_high"]["optimized_overlap"],
                  elapsed_seconds=time.monotonic() - started,
                  detector_noise_weighted=False, release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps(dict(simulation=simulation,
                          native_vs_nr_overlap=pairs["native_vs_nr"]["optimized_overlap"],
                          sources_unchanged=result["sources_unchanged"])), flush=True)
    if not result["sources_unchanged"]:
        raise ValueError("generator source changed during long-start run")


if __name__ == "__main__":
    main()
