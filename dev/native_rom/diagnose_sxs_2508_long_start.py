"""Post-hoc earlier-start q≈5 SEOBNRv4/SXS numerical-physics diagnostic.

This tests whether the late-start five-case result is stable when the native
generator begins before the archived NR reference frequency. It is not a
detector-weighted or domain-wide release test.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--long-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    primary = json.loads(args.primary.read_text())
    alignment = json.loads(args.alignment.read_text())
    long_plan = json.loads(args.long_plan.read_text())
    simulation = "SXS:BBH:2508"
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (long_plan["schema"] != "diffgw.seobnrv4.sxs-2508-long-start-plan.v1"
            or long_plan["qualification_claim"] is not False
            or long_plan["simulation"] != simulation
            or long_plan["primary_sha256"] != q.sha256(args.primary)
            or long_plan["alignment_sha256"] != q.sha256(args.alignment)
            or primary["plan_sha256"] != q.sha256(args.plan)
            or alignment["plan_sha256"] != q.sha256(args.plan)
            or primary["extraction_sha256"] != q.sha256(extraction_path)
            or not primary["all_cases_succeeded"]
            or not alignment["all_cases_succeeded"]):
        raise ValueError("long-start prerequisites changed")
    torch.set_num_threads(1)
    case = plan["cases"][simulation]
    method = plan["method"]
    row = dict(case["row"], omega_start=long_plan["omega_start"])
    dt = method["grid_step_M"]
    source_before = q.canonical_hash(q.source_snapshot())
    started = time.monotonic()
    native = q.native_run(row, dict(name="native_long_start",
        atol=method["native_atol"], rtol=method["native_rtol"], dt_geom=dt),
        q.Settings(max_run_seconds=long_plan["max_run_seconds"]),
        diagnostic_fine_rkf45_resampling=method["diagnostic_fine_rkf45_resampling"])
    public_wave = public.public_mode(row, dt, long_plan["public_call"])
    high = max(case["levels"])
    record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / record["path"]
    if q.sha256(nr_path) != record["sha256"]:
        raise ValueError("NR mode changed")
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
    result = dict(schema="diffgw.seobnrv4.sxs-2508-long-start.v1",
                  plan_sha256=q.sha256(args.plan),
                  primary_sha256=q.sha256(args.primary),
                  alignment_sha256=q.sha256(args.alignment),
                  long_plan_sha256=q.sha256(args.long_plan),
                  script_sha256=q.sha256(__file__),
                  source_manifest_sha256_before=source_before,
                  source_manifest_sha256_after=q.canonical_hash(q.source_snapshot()),
                  row=row, native=q.summarize_run(native),
                  public_call={key: public_wave[key] for key in (
                      "f_lower_Hz", "delta_t_s", "distance_Mpc", "inclination_rad")},
                  anchors={name: {key: value for key, value in wave.items()
                                  if key not in ("time", "mode")}
                           for name, wave in waves.items()},
                  array_sha256=q.sha256(args.array_out),
                  direct_native_vs_nr=c.compare(waves["native"], waves["nr"],
                        method["windows_relative_to_highest_resolution_amplitude_peak_M"],
                        dt),
                  pairs=pairs,
                  late_start_native_vs_nr_overlap=alignment["cases"][simulation]
                      ["pairs"]["native_vs_high"]["optimized_overlap"],
                  elapsed_seconds=time.monotonic()-started,
                  detector_noise_weighted=False, release_qualified=False)
    result["sources_unchanged"] = (result["source_manifest_sha256_before"]
                                   == result["source_manifest_sha256_after"])
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps(dict(native_vs_nr_overlap=pairs["native_vs_nr"]["optimized_overlap"],
                          sources_unchanged=result["sources_unchanged"])))
    if not result["sources_unchanged"]:
        raise ValueError("generator source changed during long-start run")


if __name__ == "__main__":
    main()
