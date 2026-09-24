"""Post-hoc numerical-convergence check for the failed third-holdout case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from . import compare_sxs_2123 as compare
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--weighted-result", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: json.loads(getattr(args, name).read_text()) for name in
              ("case_plan", "weighted_plan", "comparison", "weighted_result", "plan")}
    case_plan, weighted_plan, comparison, weighted_result, plan = (
        inputs[name] for name in ("case_plan", "weighted_plan", "comparison",
                                  "weighted_result", "plan"))
    simulation = "SXS:BBH:2569"
    choice_path = q.ROOT / case_plan["candidate"]["source_choice_path"]
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2569-convergence-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["simulation"] != simulation
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["weighted_plan_sha256"] != q.sha256(args.weighted_plan)
            or plan["comparison_sha256"] != q.sha256(args.comparison)
            or plan["weighted_result_sha256"] != q.sha256(args.weighted_result)
            or comparison["plan_sha256"] != q.sha256(args.case_plan)
            or comparison["all_cases_succeeded"] is not True
            or comparison["sources_unchanged"] is not True
            or weighted_result["cases"][simulation]["selected_case_99_percent_passed"] is not False
            or q.sha256(choice_path) != case_plan["candidate"]["source_choice_sha256"]
            or plan["atol"] >= case_plan["candidate"]["numerical"]["atol"]
            or plan["rtol"] >= case_plan["candidate"]["numerical"]["rtol"]
            or plan["mass_msun"] != 22):
        raise ValueError("frozen convergence prerequisites changed")
    choices = json.loads(choice_path.read_text())["choice"]
    numerical = case_plan["candidate"]["numerical"]
    entry = comparison["cases"][simulation]
    baseline_path = Path(entry["anchored_array"]["path"])
    if q.sha256(baseline_path) != entry["anchored_array"]["sha256"]:
        raise ValueError("baseline array changed")
    with np.load(baseline_path, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"], mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    import torch
    torch.set_num_threads(1)
    source_before = q.canonical_hash(q.source_snapshot())
    case = case_plan["cases"][simulation]
    dt = case_plan["method"]["grid_step_M"]
    started = time.monotonic()
    run = q.native_run(
        case["row"], dict(name="tighter", dt_geom=dt, atol=plan["atol"],
                          rtol=plan["rtol"]),
        q.Settings(max_run_seconds=plan["max_run_seconds"]),
        diagnostic_fine_rkf45_resampling=numerical["fine_rkf45_resampling"],
        diagnostic_matching_second_derivatives=numerical["matching_second_derivatives"],
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
    anchored = compare.anchor(run, dt=dt)
    waves["tighter"] = anchored
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, tighter_time=anchored["time"],
             tighter_mode=anchored["mode"])
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if q.sha256(extraction_path) != comparison["extraction_sha256"]:
        raise ValueError("NR extraction changed")
    high = max(case["levels"])
    metadata_path = (args.archive_dir / simulation.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    record = extraction["cases"][simulation]["archive_files"][f"Lev{high}:metadata.json"]
    if q.sha256(metadata_path) != record["sha256"]:
        raise ValueError("NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    settled_start = metadata["relaxation_time"] - entry["anchor"][f"Lev{high}"]["anchor_time_M"]
    f_geom, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=settled_start, dt=dt, width=weighted_plan["taper_width_M"])
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    mass = plan["mass_msun"]
    mass_seconds = mass * float(MSUN_SEC)
    hz = f_geom / mass_seconds
    lower_hz = max(weighted_plan["minimum_frequency_Hz"],
                   orbital / (math.pi * mass_seconds))
    band = (hz >= lower_hz) & (hz <= weighted_plan["maximum_frequency_Hz"])
    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                    for f in hz[band]])
    overlaps = {name: weighted.optimized_overlap(
        f_geom[band], spectra[candidate][band], spectra["nr_high"][band], psd,
        bounds=tuple(weighted_plan["allowed_time_shift_M"]),
        step=weighted_plan["time_shift_grid_step_M"])
        for name, candidate in (("baseline", "native"), ("tighter", "tighter"),
                                ("low_vs_high_nr", "nr_low"))}
    fixed = compare.compare(
        {**waves["native"], **entry["anchor"]["native"]}, anchored,
        case_plan["method"]["windows_relative_to_highest_resolution_amplitude_peak_M"],
        dt, support=entry["comparisons"]["native_vs_high"]["common_support_M"])
    difference = overlaps["tighter"]["overlap"] - overlaps["baseline"]["overlap"]
    stable = (abs(difference) <= plan["max_overlap_change"]
              and fixed["metrics"]["complex_relative_l2"]
              <= plan["max_complex_relative_l2"])
    result = dict(schema="diffgw.seobnrv4.sxs-2569-convergence.v1",
                  plan_sha256=q.sha256(args.plan), script_sha256=q.sha256(__file__),
                  case_plan_sha256=q.sha256(args.case_plan),
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  comparison_sha256=q.sha256(args.comparison),
                  weighted_result_sha256=q.sha256(args.weighted_result),
                  source_manifest_sha256_before=source_before,
                  source_manifest_sha256_after=q.canonical_hash(q.source_snapshot()),
                  baseline_array_sha256=q.sha256(baseline_path),
                  tighter_array=dict(path=str(args.array_out),
                                     sha256=q.sha256(args.array_out)),
                  extraction_sha256=q.sha256(extraction_path),
                  metadata_sha256=q.sha256(metadata_path),
                  elapsed_seconds=time.monotonic()-started,
                  native=q.summarize_run(run),
                  anchor={key: value for key, value in anchored.items()
                          if key not in ("time", "mode")},
                  fixed_baseline_vs_tighter=fixed,
                  weighted_overlaps_22_msun=overlaps,
                  overlap_change=difference,
                  numerical_gate_passed=stable,
                  frozen_physical_proxy_still_failed=(
                      overlaps["tighter"]["overlap"]
                      < weighted_plan["selected_case_99_percent_threshold"]),
                  common_time_support_M=support, fft_length=fft_length,
                  posthoc_selected_case=True, release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps(dict(numerical_gate_passed=stable,
                          overlap_change=difference,
                          tighter_overlap=overlaps["tighter"]["overlap"])), flush=True)


if __name__ == "__main__":
    main()
