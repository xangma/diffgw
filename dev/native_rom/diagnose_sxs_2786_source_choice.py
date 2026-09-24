"""Post-hoc source-backed native choice on the failed SXS:BBH:2786 holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from . import compare_sxs_2123 as c
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--intervention-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    case_plan = json.loads(args.case_plan.read_text())
    weighted_plan = json.loads(args.weighted_plan.read_text())
    comparison = json.loads(args.comparison.read_text())
    plan = json.loads(args.intervention_plan.read_text())
    choice_path = q.ROOT / plan["source_choice_path"]
    choices = json.loads(choice_path.read_text())["choice"]
    simulation = "SXS:BBH:2786"
    if (case_plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-plan.v1"
            or weighted_plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-weighted-plan.v1"
            or plan["schema"] not in {
                "diffgw.seobnrv4.sxs-2786-source-choice-plan.v1",
                "diffgw.seobnrv4.sxs-2786-source-choice-near-peak-plan.v1",
                "diffgw.seobnrv4.sxs-2786-source-choice-loose-plan.v1",
            }
            or plan["qualification_claim"] is not False
            or plan["simulation"] != simulation
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["weighted_plan_sha256"] != q.sha256(args.weighted_plan)
            or plan["comparison_sha256"] != q.sha256(args.comparison)
            or plan["source_choice_sha256"] != q.sha256(choice_path)
            or comparison["all_cases_succeeded"] is not True
            or not comparison["sources_unchanged"]
            or comparison["plan_sha256"] != q.sha256(args.case_plan)):
        raise ValueError("source-choice intervention prerequisites changed")
    numerical = plan["numerical_changes"]
    if (numerical["fine_rkf45_resampling"] is not True
            or numerical["warm_start_Momega"] is not None
            or choices["nqc_phase_window_M"] != numerical["nqc_phase_window_M"]
            or (numerical["restart_time_M"] is not None
                and choices["numerical"]["fixed_restart_time_M"]
                != numerical["restart_time_M"])
            or (numerical["restart_time_M"] is None
                and plan["schema"] not in {
                    "diffgw.seobnrv4.sxs-2786-source-choice-near-peak-plan.v1",
                    "diffgw.seobnrv4.sxs-2786-source-choice-loose-plan.v1",
                })):
        raise ValueError("intervention numerical choices inconsistent")
    loose = plan["schema"] == "diffgw.seobnrv4.sxs-2786-source-choice-loose-plan.v1"
    if loose and (plan["atol"] < case_plan["method"]["native_atol"]
                  or plan["rtol"] < case_plan["method"]["native_rtol"]
                  or not plan["trace_stages"]):
        raise ValueError("loose diagnostic settings inconsistent")
    import torch
    torch.set_num_threads(1)
    source_before = q.canonical_hash(q.source_snapshot())
    case = case_plan["cases"][simulation]
    dt = case_plan["method"]["grid_step_M"]
    start = time.monotonic()
    stage_trace = []
    if plan.get("trace_stages", False):
        from .eob.torch import adaptive_driver_clean as driver
        original_integrate = driver.integrate_adaptive

        def traced_integrate(*args, **kwargs):
            stage = ("coarse", "fine")[min(len(stage_trace), 1)]
            print(f"{stage} integration started", flush=True)
            stage_start = time.monotonic()
            result = original_integrate(*args, **kwargs)
            record = dict(stage=stage, elapsed_seconds=time.monotonic()-stage_start,
                          accepted=result.statistics.total_accepted,
                          rejected=result.statistics.total_rejected,
                          evaluations=result.statistics.total_evaluations)
            stage_trace.append(record)
            print(json.dumps(record), flush=True)
            return result

        driver.integrate_adaptive = traced_integrate
    try:
        run = q.native_run(
            case["row"], dict(name="source_choice", dt_geom=dt,
                              atol=plan.get("atol", case_plan["method"]["native_atol"]),
                              rtol=plan.get("rtol", case_plan["method"]["native_rtol"])),
            q.Settings(max_run_seconds=plan["max_run_seconds"]),
            diagnostic_fine_rkf45_resampling=True,
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
    finally:
        if plan.get("trace_stages", False):
            driver.integrate_adaptive = original_integrate
    elapsed = time.monotonic() - start
    anchored = c.anchor(run, dt=dt)
    primary = comparison["cases"][simulation]
    original_path = Path(primary["anchored_array"]["path"])
    if q.sha256(original_path) != primary["anchored_array"]["sha256"]:
        raise ValueError("primary holdout array changed")
    with np.load(original_path, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"],
                            mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    waves["source_choice"] = anchored
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, source_choice_time=anchored["time"],
             source_choice_mode=anchored["mode"])
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (q.sha256(extraction_path) != comparison["extraction_sha256"]
            or extraction["plan_sha256"] != q.sha256(args.case_plan)):
        raise ValueError("NR extraction changed")
    high = max(case["levels"])
    metadata_path = (args.archive_dir / simulation.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    record = extraction["cases"][simulation]["archive_files"][f"Lev{high}:metadata.json"]
    if q.sha256(metadata_path) != record["sha256"]:
        raise ValueError("NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    settled_start = metadata["relaxation_time"] - primary["anchor"][f"Lev{high}"]["anchor_time_M"]
    f_geom, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=settled_start, dt=dt, width=weighted_plan["taper_width_M"])
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    rows = []
    for mass in plan["masses_msun"]:
        mass_seconds = mass * float(MSUN_SEC)
        hz = f_geom / mass_seconds
        low_hz = max(weighted_plan["minimum_frequency_Hz"],
                     orbital / (math.pi * mass_seconds))
        band = (hz >= low_hz) & (hz <= weighted_plan["maximum_frequency_Hz"])
        psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                        for f in hz[band]])
        results = {name: weighted.optimized_overlap(
            f_geom[band], spectra[candidate][band], spectra["nr_high"][band], psd,
            bounds=tuple(weighted_plan["allowed_time_shift_M"]),
            step=weighted_plan["time_shift_grid_step_M"])
            for name, candidate in (("default_native_vs_nr", "native"),
                                    ("source_choice_vs_nr", "source_choice"),
                                    ("low_vs_high_nr", "nr_low"))}
        rows.append(dict(total_mass_msun=mass, lower_frequency_Hz=low_hz,
                         frequency_bins=int(np.count_nonzero(band)), results=results))
    nr_record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / nr_record["path"]
    if q.sha256(nr_path) != nr_record["sha256"]:
        raise ValueError("NR mode changed")
    nr = c.anchor(c.load_mode(nr_path), dt=dt)
    fixed = c.compare(anchored, nr,
                      case_plan["method"]["windows_relative_to_highest_resolution_amplitude_peak_M"],
                      dt, support=primary["comparisons"]["native_vs_high"]["common_support_M"])
    result = dict(schema="diffgw.seobnrv4.sxs-2786-source-choice.v1",
                  plan_sha256=q.sha256(args.intervention_plan),
                  script_sha256=q.sha256(__file__),
                  source_manifest_sha256_before=source_before,
                  source_manifest_sha256_after=q.canonical_hash(q.source_snapshot()),
                  case_plan_sha256=q.sha256(args.case_plan),
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  comparison_sha256=q.sha256(args.comparison),
                  source_choice_sha256=q.sha256(choice_path),
                  extraction_sha256=q.sha256(extraction_path),
                  original_array_sha256=q.sha256(original_path),
                  stage_trace=stage_trace,
                  array=dict(path=str(args.array_out), sha256=q.sha256(args.array_out)),
                  elapsed_seconds=elapsed, native=q.summarize_run(run),
                  anchor={key: value for key, value in anchored.items()
                          if key not in ("time", "mode")},
                  fixed_crossing_vs_nr=fixed,
                  common_time_support_M=support, fft_length=fft_length,
                  rows=rows, posthoc_selected_case=True,
                  release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps({row["total_mass_msun"]: {
        name: item["overlap"] for name, item in row["results"].items()}
        for row in rows}), flush=True)


if __name__ == "__main__":
    main()
