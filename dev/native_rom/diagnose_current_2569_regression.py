"""Re-evaluate the known SXS:BBH:2569 failure on the current source snapshot.

This is a development regression diagnostic on an already inspected NR case,
not an untouched holdout or release qualification.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch

from . import compare_sxs_2123 as comparison
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--array-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["schema"] != "diffgw.seobnrv4.known-2569-current-regression-plan.v1":
        raise ValueError("unexpected regression plan")
    case_plan_path = q.ROOT / plan["case_plan"]
    weighted_plan_path = q.ROOT / plan["weighted_plan"]
    previous_path = q.ROOT / plan["previous_comparison"]
    if (q.sha256(case_plan_path) != plan["case_plan_sha256"]
            or q.sha256(weighted_plan_path) != plan["weighted_plan_sha256"]
            or q.sha256(previous_path) != plan["previous_comparison_sha256"]
            or q.canonical_hash(q.source_snapshot()) != plan["source_snapshot_sha256"]):
        raise ValueError("a frozen plan input or current source changed")
    cases = json.loads(case_plan_path.read_text())
    gate = json.loads(weighted_plan_path.read_text())
    previous = json.loads(previous_path.read_text())
    simulation = plan["simulation"]
    case = cases["cases"][simulation]
    old = previous["cases"][simulation]
    if (simulation != "SXS:BBH:2569"
            or previous["plan_sha256"] != q.sha256(case_plan_path)
            or gate["case_plan_sha256"] != q.sha256(case_plan_path)
            or gate["selected_case_99_percent_threshold"] != 0.99
            or gate["mass_grid_msun"] != {"first": 10, "last": 200, "step": 1}
            or q.sha256(old["anchored_array"]["path"])
            != old["anchored_array"]["sha256"]):
        raise ValueError("the known-case evidence or gate changed")
    choices_path = q.ROOT / cases["candidate"]["source_choice_path"]
    if q.sha256(choices_path) != cases["candidate"]["source_choice_sha256"]:
        raise ValueError("candidate choices changed")
    choices = json.loads(choices_path.read_text())["choice"]
    numerical = cases["candidate"]["numerical"]
    method = cases["method"]
    torch.set_num_threads(1)
    started = time.monotonic()
    native = q.native_run(
        case["row"], dict(name="source_candidate", atol=method["native_atol"],
                          rtol=method["native_rtol"], dt_geom=method["grid_step_M"]),
        q.Settings(max_run_seconds=method.get("max_run_seconds_per_case", 600.)),
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
        diagnostic_hamiltonian_calibration=choices["calibration"],
    )
    current = comparison.anchor(native, dt=method["grid_step_M"])
    with np.load(old["anchored_array"]["path"], allow_pickle=False) as source:
        previous_native = dict(time=source["native_time"], mode=source["native_mode"])
        nr_low = dict(time=source["nr_low_time"], mode=source["nr_low_mode"])
        nr_high = dict(time=source["nr_high_time"], mode=source["nr_high_mode"])
    args.array_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.array_out, native_time=current["time"], native_mode=current["mode"],
             nr_low_time=nr_low["time"], nr_low_mode=nr_low["mode"],
             nr_high_time=nr_high["time"], nr_high_mode=nr_high["mode"])

    high = max(case["levels"])
    metadata_path = (Path(plan["archive_dir"]) / simulation.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    if q.sha256(metadata_path) != plan["nr_high_metadata_sha256"]:
        raise ValueError("NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    settled_start = metadata["relaxation_time"] - old["anchor"][f"Lev{high}"]["anchor_time_M"]
    if (metadata["reference_time"] != metadata["relaxation_time"]
            or not math.isclose(orbital, float(np.linalg.norm(
                case["reference_orbital_frequency"])), abs_tol=1e-12)
            or current["time"][0] > settled_start + gate["sample_step_M"]):
        raise ValueError("current waveform does not cover settled NR reference")
    waves = {"native": current, "nr_low": nr_low, "nr_high": nr_high}
    frequencies, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=settled_start, dt=gate["sample_step_M"],
        width=gate["taper_width_M"])
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    q.lal_binary_identity(gate["binary_versions"]["lal"],
                          gate["binary_versions"]["lalsimulation"])
    scores = []
    for mass in range(gate["mass_grid_msun"]["first"],
                      gate["mass_grid_msun"]["last"] + 1,
                      gate["mass_grid_msun"]["step"]):
        frequency_hz = frequencies / (mass * float(MSUN_SEC))
        lower_hz = max(gate["minimum_frequency_Hz"],
                       orbital / (math.pi * mass * float(MSUN_SEC)))
        band = ((frequency_hz >= lower_hz)
                & (frequency_hz <= gate["maximum_frequency_Hz"]))
        if np.count_nonzero(band) < 8:
            raise ValueError(f"insufficient detector band at {mass} solar masses")
        psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                        for f in frequency_hz[band]])
        score = weighted.optimized_overlap(
            frequencies[band], spectra["native"][band], spectra["nr_high"][band], psd,
            bounds=tuple(gate["allowed_time_shift_M"]),
            step=gate["time_shift_grid_step_M"])
        scores.append(dict(total_mass_msun=mass, lower_frequency_Hz=lower_hz,
                           frequency_bins=int(np.count_nonzero(band)), **score))
    minimum = min(scores, key=lambda score: score["overlap"])
    windows = method["windows_relative_to_highest_resolution_amplitude_peak_M"]
    result = dict(schema="diffgw.seobnrv4.known-2569-current-regression.v1",
                  plan_sha256=q.sha256(args.plan), script_sha256=q.sha256(__file__),
                  current_source_snapshot_sha256=plan["source_snapshot_sha256"],
                  source_snapshot_sha256_after=q.canonical_hash(q.source_snapshot()),
                  old_array_sha256=old["anchored_array"]["sha256"],
                  current_array_sha256=q.sha256(args.array_out),
                  nr_high_metadata_sha256=q.sha256(metadata_path),
                  process=dict(host=socket.gethostname(), elapsed_seconds=time.monotonic()-started),
                  numerical_choice=numerical, row=case["row"],
                  current_vs_previous=comparison.compare(current, previous_native, windows,
                                                         method["grid_step_M"]),
                  current_vs_nr=comparison.compare(current, nr_high, windows,
                                                   method["grid_step_M"]),
                  nr_settled_start_after_anchor_M=settled_start,
                  weighted=dict(common_time_support_M=support, fft_length=fft_length,
                                minimum=minimum, all_masses=scores,
                                known_case_99_percent_passed=minimum["overlap"] >= 0.99),
                  release_qualified=False, untouched_holdout=False)
    if result["source_snapshot_sha256_after"] != plan["source_snapshot_sha256"]:
        raise ValueError("generator source changed during the run")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps(dict(minimum=minimum, known_case_99_percent_passed=
                          result["weighted"]["known_case_99_percent_passed"],
                          elapsed_seconds=result["process"]["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
