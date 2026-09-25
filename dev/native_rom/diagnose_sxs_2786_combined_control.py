"""Reproduce the frozen post-hoc SXS:BBH:2786 combined-control diagnostic."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as comparison_tools
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


EVIDENCE = q.ROOT / "dev/native_rom/evidence"
PRIOR = EVIDENCE / "2026-09-24"
CURRENT = EVIDENCE / "2026-09-25"
PLAN = CURRENT / "sxs_2786_combined_control_plan.json"
OUTPUT = CURRENT / "sxs_2786_combined_control.json"
DECODED = Path("/tmp/diffgw-sxs-second-holdout/decoded")
ARCHIVE = Path("/tmp/diffgw-sxs-second-holdout/archive")
ARRAY = DECODED / "SXS_BBH_2786_combined_control.npz"
SIMULATION = "SXS:BBH:2786"


def _inputs(plan):
    names = {
        "case_plan": "sxs_second_holdout_plan.json",
        "weighted_plan": "sxs_second_holdout_weighted_plan.json",
        "comparison": "sxs_second_holdout_comparison.json",
        "prior_intervention": "sxs_2786_source_choice_optimized.json",
        "source_choice": "source_candidate_new_rows.json",
    }
    records = {}
    for name, basename in names.items():
        path = PRIOR / basename
        if q.sha256(path) != plan[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} changed")
        records[name] = json.loads(path.read_text())
    original = DECODED / "SXS_BBH_2786_anchored.npz"
    earlier = DECODED / "SXS_BBH_2786_source_choice_optimized.npz"
    extraction_path = DECODED / "extraction.json"
    for name, path in (("original_array", original),
                       ("prior_intervention_array", earlier),
                       ("extraction", extraction_path)):
        if q.sha256(path) != plan[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} changed")
    records["extraction"] = json.loads(extraction_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2786-combined-control-plan.v1"
            or plan["release_qualified"] is not False
            or plan["simulation"] != SIMULATION
            or plan["masses_msun"] != [10, 50, 58, 100, 200]
            or plan["selected_case_overlap_threshold"] != 0.99
            or records["weighted_plan"]["selected_case_99_percent_threshold"] != 0.99
            or records["comparison"]["plan_sha256"] != plan["case_plan_sha256"]
            or records["comparison"]["all_cases_succeeded"] is not True
            or records["extraction"]["plan_sha256"] != plan["case_plan_sha256"]
            or records["source_choice"]["choice"]["calibration"] != "lal_v4_rounded"):
        raise ValueError("frozen physical comparison prerequisites changed")
    return records, original, earlier


def main():
    import torch
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    plan = json.loads(PLAN.read_text())
    records, original_path, earlier_path = _inputs(plan)
    torch.set_num_threads(1)
    before = q.canonical_hash(q.source_snapshot())
    started = time.monotonic()
    result = dict(schema="diffgw.seobnrv4.sxs-2786-combined-control.v1",
                  status="running", release_qualified=False,
                  posthoc_selected_case=True, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), source_sha256_before=before,
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        case = records["case_plan"]["cases"][SIMULATION]
        choice = records["source_choice"]["choice"]
        numerical = plan["numerical"]
        print("combined-control native integration started", flush=True)
        native = q.native_run(
            case["row"], dict(name="combined_control", dt_geom=numerical["output_step_M"],
                              atol=numerical["atol"], rtol=numerical["rtol"]),
            q.Settings(max_run_seconds=numerical["max_run_seconds"]),
            diagnostic_spin_truncation=choice["spin_truncation"],
            diagnostic_mode_21_higher_spin=choice["flux"]["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=choice["flux"]["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=choice["flux"]["mode_32_spin_square"],
            diagnostic_mode_33_higher_spin=choice["physical"]["mode_33_higher_spin"],
            diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
            diagnostic_hamiltonian_log_derivative=
                choice["physical"]["hamiltonian_log_derivative"],
            diagnostic_hamiltonian_calibration=choice["calibration"],
            diagnostic_qnm_source=numerical["qnm_source"],
            diagnostic_coarse_initial_step_M=numerical["coarse_initial_step_M"],
            diagnostic_fine_max_step_M=numerical["fine_max_step_M"],
            diagnostic_fine_refinement_factor=numerical["fine_refinement_factor"],
            diagnostic_combined_local_fit=numerical["combined_local_fit"],
            return_nqc_coefficients=True)
        print("combined-control native integration finished", flush=True)
        dt = numerical["output_step_M"]
        anchored = comparison_tools.anchor(native, dt=dt)
        with np.load(original_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"])
                     for name in ("native", "nr_low", "nr_high")}
        with np.load(earlier_path, allow_pickle=False) as source:
            waves["earlier_choice"] = dict(time=source["source_choice_time"],
                                           mode=source["source_choice_mode"])
        waves["combined_control"] = anchored
        np.savez(ARRAY, combined_control_time=anchored["time"],
                 combined_control_mode=anchored["mode"])
        case_record = records["comparison"]["cases"][SIMULATION]
        high = max(case["levels"])
        metadata_path = (ARCHIVE / SIMULATION.replace(":", "_")
                         / f"Lev{high}" / "metadata.json")
        metadata_record = records["extraction"]["cases"][SIMULATION]
        metadata_record = metadata_record["archive_files"][f"Lev{high}:metadata.json"]
        if q.sha256(metadata_path) != metadata_record["sha256"]:
            raise ValueError("NR metadata changed")
        metadata = json.loads(metadata_path.read_text())
        orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
        settled = (metadata["relaxation_time"]
                   - case_record["anchor"][f"Lev{high}"]["anchor_time_M"])
        weight_plan = records["weighted_plan"]
        frequency, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=settled, dt=dt, width=weight_plan["taper_width_M"])
        rows = []
        for mass in plan["masses_msun"]:
            mass_seconds = mass * float(MSUN_SEC)
            hz = frequency / mass_seconds
            low_hz = max(weight_plan["minimum_frequency_Hz"],
                         orbital / (math.pi * mass_seconds))
            band = (hz >= low_hz) & (hz <= weight_plan["maximum_frequency_Hz"])
            psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                            for f in hz[band]])
            scores = {name: weighted.optimized_overlap(
                frequency[band], spectra[name][band], spectra["nr_high"][band],
                psd, bounds=tuple(weight_plan["allowed_time_shift_M"]),
                step=weight_plan["time_shift_grid_step_M"])
                for name in ("combined_control", "native", "earlier_choice", "nr_low")}
            rows.append(dict(total_mass_msun=mass, lower_frequency_Hz=low_hz,
                             frequency_bins=int(np.count_nonzero(band)), scores=scores))
        values = [row["scores"]["combined_control"]["overlap"] for row in rows]
        result.update(status="diagnostic_complete", native=q.summarize_run(native),
                      anchor={key: value for key, value in anchored.items()
                              if key not in ("time", "mode")},
                      array=dict(path=str(ARRAY), sha256=q.sha256(ARRAY)),
                      common_time_support_M=support, fft_length=fft_length,
                      rows=rows, minimum_overlap=min(values),
                      selected_case_99_percent_passed=all(
                          value >= plan["selected_case_overlap_threshold"]
                          for value in values))
    except Exception as exc:
        result.update(status="failed", error=dict(type=type(exc).__name__,
                                                   message=str(exc)))
    result["source_sha256_after"] = q.canonical_hash(q.source_snapshot())
    result["sources_unchanged"] = result["source_sha256_after"] == before
    result["elapsed_s"] = time.monotonic() - started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"],
                          minimum_overlap=result.get("minimum_overlap"),
                          error=result.get("error"))), flush=True)
    if result["status"] != "diagnostic_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
