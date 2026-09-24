"""Post-hoc pinned-public waveform parity on the third SXS holdout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as compare
from . import compare_sxs_2123_public_control as public
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
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    documents = {key: json.loads(getattr(args, key).read_text()) for key in
                 ("case_plan", "weighted_plan", "comparison", "weighted_result", "plan")}
    case_plan, weighted_plan, comparison, weighted_result, plan = (
        documents[key] for key in ("case_plan", "weighted_plan", "comparison",
                                   "weighted_result", "plan"))
    if (plan["schema"] != "diffgw.seobnrv4.sxs-third-public-parity-plan.v1"
            or plan["qualification_claim"] is not False
            or any(plan[f"{key}_sha256"] != q.sha256(getattr(args, key)) for key in
                   ("case_plan", "weighted_plan", "comparison", "weighted_result"))
            or comparison["all_cases_succeeded"] is not True
            or comparison["sources_unchanged"] is not True
            or weighted_result["all_cases_succeeded"] is not True
            or set(comparison["cases"]) != set(case_plan["cases"])):
        raise ValueError("third holdout or parity plan changed")
    binary = q.lal_binary_identity(plan["binary_versions"]["lal"],
                                   plan["binary_versions"]["lalsimulation"])
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if q.sha256(extraction_path) != comparison["extraction_sha256"]:
        raise ValueError("NR extraction changed")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    dt = case_plan["method"]["grid_step_M"]
    windows = case_plan["method"][
        "windows_relative_to_highest_resolution_amplitude_peak_M"]
    cases = {}
    for simulation, case in case_plan["cases"].items():
        entry = comparison["cases"][simulation]
        array_path = Path(entry["anchored_array"]["path"])
        if q.sha256(array_path) != entry["anchored_array"]["sha256"]:
            raise ValueError(f"{simulation} anchored array changed")
        with np.load(array_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"])
                     for name in ("native", "nr_low", "nr_high")}
        public_wave = compare.anchor(public.public_mode(case["row"], dt,
                                                          plan["public_call"]), dt=dt)
        waves["public"] = public_wave
        fixed = compare.compare(
            {**waves["native"], **entry["anchor"]["native"]}, public_wave,
            windows, dt, support=entry["comparisons"]["native_vs_high"][
                "common_support_M"])
        high = max(case["levels"])
        metadata_path = (args.archive_dir / simulation.replace(":", "_")
                         / f"Lev{high}" / "metadata.json")
        record = extraction["cases"][simulation]["archive_files"][f"Lev{high}:metadata.json"]
        if q.sha256(metadata_path) != record["sha256"]:
            raise ValueError(f"{simulation} NR metadata changed")
        metadata = json.loads(metadata_path.read_text())
        orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
        settled_start = (metadata["relaxation_time"]
                         - entry["anchor"][f"Lev{high}"]["anchor_time_M"])
        f_geom, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=settled_start, dt=dt,
            width=weighted_plan["taper_width_M"])
        mass = weighted_result["cases"][simulation]["mass_at_minimum_msun"]
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
            for name, candidate in (("native_vs_nr", "native"),
                                    ("public_vs_nr", "public"),
                                    ("low_vs_high_nr", "nr_low"))}
        delta = (overlaps["native_vs_nr"]["overlap"]
                 - overlaps["public_vs_nr"]["overlap"])
        cases[simulation] = dict(
            anchored_array_sha256=q.sha256(array_path),
            nr_metadata_sha256=q.sha256(metadata_path),
            public_anchor={key: value for key, value in public_wave.items()
                           if key not in ("time", "mode")},
            fixed_native_vs_public=fixed,
            mass_at_selected_minimum_msun=mass,
            weighted_overlaps_at_selected_minimum=overlaps,
            native_minus_public_weighted_overlap=delta,
            fixed_gate_passed=(fixed["metrics"]["complex_relative_l2"]
                               <= plan["max_fixed_crossing_complex_relative_l2"]),
            weighted_gate_passed=(abs(delta) <= plan[
                "max_minimum_mass_weighted_overlap_difference"]),
            common_time_support_M=support, fft_length=fft_length)
        print(json.dumps(dict(simulation=simulation,
                              fixed_complex_relative_l2=fixed["metrics"][
                                  "complex_relative_l2"],
                              weighted_overlap_difference=delta)), flush=True)
    result = dict(schema="diffgw.seobnrv4.sxs-third-public-parity.v1",
                  plan_sha256=q.sha256(args.plan),
                  case_plan_sha256=q.sha256(args.case_plan),
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  comparison_sha256=q.sha256(args.comparison),
                  weighted_result_sha256=q.sha256(args.weighted_result),
                  extraction_sha256=q.sha256(extraction_path),
                  script_sha256=q.sha256(__file__), lal_binary=binary,
                  posthoc_selected_cases=True, release_qualified=False,
                  all_frozen_parity_gates_passed=all(
                      item["fixed_gate_passed"] and item["weighted_gate_passed"]
                      for item in cases.values()), cases=cases)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
