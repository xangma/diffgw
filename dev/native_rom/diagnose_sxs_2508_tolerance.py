"""Frozen q≈5 long-start ODE tolerance sensitivity; development evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import compare_sxs_2123 as c
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--long-plan", type=Path, required=True)
    parser.add_argument("--long-result", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--baseline-array", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    baseline = json.loads(args.long_result.read_text())
    long_plan = json.loads(args.long_plan.read_text())
    weight_plan = json.loads(args.weighted_plan.read_text())
    levels = plan["levels"]
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2508-tolerance-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["simulation"] != "SXS:BBH:2508"
            or baseline["schema"] != "diffgw.seobnrv4.sxs-2508-long-start.v1"
            or baseline["long_plan_sha256"] != q.sha256(args.long_plan)
            or baseline["array_sha256"] != q.sha256(args.baseline_array)
            or q.sha256(args.long_result) != plan["long_result_sha256"]
            or q.sha256(args.long_plan) != plan["long_plan_sha256"]
            or q.sha256(args.weighted_plan) != plan["weighted_plan_sha256"]
            or q.sha256(args.baseline_array) != plan["baseline_array_sha256"]
            or baseline["source_manifest_sha256_after"]
            != plan["baseline_source_manifest_sha256"]
            or not baseline["sources_unchanged"]
            or weight_plan["sample_step_M"] != plan["grid_step_M"]
            or weight_plan["taper_width_M"] != plan["weighted_taper_width_M"]
            or long_plan["omega_start"] != baseline["row"]["omega_start"]
            or [level["name"] for level in levels] != ["loose", "medium", "baseline"]
            or [(level["atol"], level["rtol"]) for level in levels]
            != [(1e-13, 1e-12), (1e-14, 1e-13), (1e-15, 1e-14)]
            or (baseline["native"]["atol"], baseline["native"]["rtol"])
            != (levels[-1]["atol"], levels[-1]["rtol"])):
        raise ValueError("frozen q≈5 tolerance prerequisites changed")
    source_before = q.canonical_hash(q.source_snapshot())
    if source_before != plan["baseline_source_manifest_sha256"]:
        raise ValueError("native generator changed since baseline")
    with np.load(args.baseline_array, allow_pickle=False) as arrays:
        anchored = {name: dict(time=arrays[f"{name}_time"],
                               mode=arrays[f"{name}_mode"])
                    for name in ("native", "nr")}
    anchor = baseline["anchors"]["native"]
    raw = dict(time=anchored["native"]["time"] + anchor["anchor_time_M"],
               mode=anchored["native"]["mode"] * np.exp(1j * anchor["anchor_phase_rad"]),
               **{key: baseline["native"][key] for key in (
                   "dt_geom", "peak_time_M", "match_time_M", "boundary_time_M")})
    if (not np.allclose(raw["time"], np.arange(len(raw["time"])) * plan["grid_step_M"],
                        rtol=0, atol=1e-9)
            or len(raw["time"]) != baseline["native"]["samples"]):
        raise ValueError("baseline raw-grid reconstruction failed")

    import lalsimulation as lalsim
    import torch
    from torchwave.constants import MSUN_SEC

    q.lal_binary_identity(weight_plan["binary_versions"]["lal"],
                          weight_plan["binary_versions"]["lalsimulation"])
    torch.set_num_threads(1)
    settings = q.Settings(max_run_seconds=plan["max_run_seconds_per_level"])
    runs = {}
    started = time.monotonic()
    for level in levels[:-1]:
        print(f"starting {level['name']} tolerance run", flush=True)
        runs[level["name"]] = q.native_run(
            baseline["row"], dict(level, dt_geom=plan["grid_step_M"]), settings,
            diagnostic_fine_rkf45_resampling=plan["diagnostic_fine_rkf45_resampling"])
        print(f"finished {level['name']} in {time.monotonic()-started:.1f} s", flush=True)
    runs["baseline"] = raw
    direct = {name: q.compare_runs(runs[name], raw) for name in ("loose", "medium")}
    trend = q.compare_runs(runs["loose"], runs["medium"])
    anchors = {name: c.anchor(runs[name], dt=plan["grid_step_M"])
               for name in ("loose", "medium")}
    waves = dict(anchors, baseline=anchored["native"], nr=anchored["nr"])
    # The settled NR support and PSD convention are already frozen by the weighted plan.
    nr_relaxed_start = (weight_plan["nr_relaxation_time_M"]
                        - baseline["anchors"]["nr"]["anchor_time_M"])
    freq, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=nr_relaxed_start, dt=plan["grid_step_M"],
        width=plan["weighted_taper_width_M"])
    mass_seconds = plan["weighted_mass_msun"] * float(MSUN_SEC)
    hz = freq / mass_seconds
    lower_hz = max(weight_plan["minimum_frequency_Hz"],
                   weight_plan["nr_reference_orbital_MOmega"] / (np.pi * mass_seconds))
    band = (hz >= lower_hz) & (hz <= weight_plan["maximum_frequency_Hz"])
    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                    for f in hz[band]])
    weighted_rows = {name: weighted.optimized_overlap(
        freq[band], spectra[name][band], spectra["nr"][band], psd,
        bounds=tuple(weight_plan["allowed_time_shift_M"]),
        step=weight_plan["time_shift_grid_step_M"])
        for name in ("loose", "medium", "baseline")}
    weighted_rows["loose_vs_baseline"] = weighted.optimized_overlap(
        freq[band], spectra["loose"][band], spectra["baseline"][band], psd,
        bounds=tuple(weight_plan["allowed_time_shift_M"]),
        step=weight_plan["time_shift_grid_step_M"])
    weighted_rows["medium_vs_baseline"] = weighted.optimized_overlap(
        freq[band], spectra["medium"][band], spectra["baseline"][band], psd,
        bounds=tuple(weight_plan["allowed_time_shift_M"]),
        step=weight_plan["time_shift_grid_step_M"])
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, **{
        f"{name}_{key}": run[key] for name, run in runs.items()
        if name != "baseline" for key in ("time", "mode")})
    source_after = q.canonical_hash(q.source_snapshot())
    result = dict(schema="diffgw.seobnrv4.sxs-2508-tolerance.v1",
                  plan_sha256=q.sha256(args.plan), long_result_sha256=q.sha256(args.long_result),
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  baseline_array_sha256=q.sha256(args.baseline_array),
                  script_sha256=q.sha256(__file__), array_sha256=q.sha256(args.array_out),
                  source_manifest_sha256_before=source_before,
                  source_manifest_sha256_after=source_after,
                  sources_unchanged=source_before == source_after,
                  levels={name: q.summarize_run(run) for name, run in runs.items()
                          if name != "baseline"},
                  direct_to_baseline=direct, loose_to_medium=trend,
                  weighted_mass_msun=plan["weighted_mass_msun"],
                  weighted_lower_frequency_Hz=lower_hz,
                  weighted_support_M=support, weighted_fft_length=fft_length,
                  weighted_bins=int(np.count_nonzero(band)), weighted=weighted_rows,
                  anchored_peak_time_differences_M={
                      name: anchors[name]["peak_after_anchor_M"]
                      - baseline["anchors"]["native"]["peak_after_anchor_M"]
                      for name in anchors},
                  direct_budget_passed=all(
                      direct["medium"][key] <= budget for key, budget in (
                          ("complex_relative_l2", plan["direct_complex_relative_l2_budget"]),
                          ("peak_time_M_abs_difference", plan["event_time_difference_budget_M"]),
                          ("match_time_M_abs_difference", plan["event_time_difference_budget_M"]),
                          ("boundary_time_M_abs_difference", plan["event_time_difference_budget_M"]))),
                  monotone_direct_error=direct["medium"]["complex_relative_l2"]
                  <= trend["complex_relative_l2"] + 1e-12,
                  elapsed_seconds=time.monotonic() - started,
                  release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps(dict(direct_budget_passed=result["direct_budget_passed"],
                          monotone_direct_error=result["monotone_direct_error"],
                          medium_l2=direct["medium"]["complex_relative_l2"],
                          weighted=weighted_rows), flush=True))
    if not result["sources_unchanged"]:
        raise ValueError("native generator changed during tolerance run")


if __name__ == "__main__":
    main()
