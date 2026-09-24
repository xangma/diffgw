"""Prospectively frozen five-case detector-weighted h22 holdout diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    case_plan = json.loads(args.case_plan.read_text())
    plan = json.loads(args.weighted_plan.read_text())
    comparison = json.loads(args.comparison.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    mass_grid = plan["mass_grid_msun"]
    if (case_plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-plan.v1"
            or plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-weighted-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["weighted_function_source_sha256"] != q.sha256(weighted.__file__)
            or extraction["schema"] != "diffgw.seobnrv4.sxs-second-holdout-extraction.v1"
            or extraction["plan_sha256"] != q.sha256(args.case_plan)
            or comparison["schema"] != "diffgw.seobnrv4.sxs-second-holdout-comparison.v1"
            or comparison["plan_sha256"] != q.sha256(args.case_plan)
            or comparison["extraction_sha256"] != q.sha256(extraction_path)
            or not comparison["all_cases_succeeded"]
            or not comparison["sources_unchanged"]
            or set(comparison["cases"]) != set(case_plan["cases"])
            or set(extraction["cases"]) != set(case_plan["cases"])
            or mass_grid != {"first": 10, "last": 200, "step": 1}
            or plan["pairs"] != ["native_vs_nr", "low_vs_high_nr"]
            or plan["sample_step_M"] != case_plan["method"]["grid_step_M"]):
        raise ValueError("second holdout weighted prerequisites changed")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    binary = q.lal_binary_identity(plan["binary_versions"]["lal"],
                                   plan["binary_versions"]["lalsimulation"])
    masses = list(range(mass_grid["first"], mass_grid["last"] + 1,
                        mass_grid["step"]))
    result = dict(schema="diffgw.seobnrv4.sxs-second-holdout-weighted.v1",
                  plan_sha256=q.sha256(args.weighted_plan),
                  case_plan_sha256=q.sha256(args.case_plan),
                  extraction_sha256=q.sha256(extraction_path),
                  comparison_sha256=q.sha256(args.comparison),
                  script_sha256=q.sha256(__file__), lal_binary=binary,
                  detector_noise_weighted=True, dominant_mode_proxy=True,
                  prospectively_fixed_after_prior_exploration=True,
                  release_qualified=False, cases={})
    for simulation, case in case_plan["cases"].items():
        try:
            entry = comparison["cases"][simulation]
            array = entry["anchored_array"]
            array_path = Path(array["path"])
            low, high = case["levels"]
            high_metadata_path = (args.archive_dir / simulation.replace(":", "_")
                                  / f"Lev{high}" / "metadata.json")
            metadata_record = extraction["cases"][simulation]["archive_files"][
                f"Lev{high}:metadata.json"]
            if (q.sha256(array_path) != array["sha256"]
                    or q.sha256(high_metadata_path) != metadata_record["sha256"]):
                raise ValueError("anchored array or NR metadata changed")
            metadata = json.loads(high_metadata_path.read_text())
            orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
            if (metadata["reference_time"] != metadata["relaxation_time"]
                    or metadata["relaxation_time"] != case["relaxation_time_M"]
                    or not math.isclose(orbital, float(np.linalg.norm(
                        case["reference_orbital_frequency"])), abs_tol=1e-12)):
                raise ValueError("NR settled-frequency definition changed")
            with np.load(array_path, allow_pickle=False) as source:
                waves = {name: dict(time=source[f"{name}_time"],
                                    mode=source[f"{name}_mode"])
                         for name in ("native", "nr_low", "nr_high")}
            settled_start = (metadata["relaxation_time"]
                             - entry["anchor"][f"Lev{high}"]["anchor_time_M"])
            late = {name: float(wave["time"][0]) for name, wave in waves.items()
                    if float(wave["time"][0]) > settled_start + plan["sample_step_M"]}
            if late:
                raise ValueError(f"waveform begins after settled NR reference: {late}")

            def sweep(width: float, selected_masses: list[int]) -> dict:
                frequency_geom, spectra, support, fft_length = weighted.tapered_spectra(
                    waves, lower=settled_start, dt=plan["sample_step_M"], width=width)
                rows = []
                for mass in selected_masses:
                    mass_seconds = mass * float(MSUN_SEC)
                    frequency_hz = frequency_geom / mass_seconds
                    lower_hz = max(plan["minimum_frequency_Hz"],
                                   orbital / (math.pi * mass_seconds))
                    band = ((frequency_hz >= lower_hz)
                            & (frequency_hz <= plan["maximum_frequency_Hz"]))
                    if np.count_nonzero(band) < 8:
                        raise ValueError(f"insufficient detector band at {mass} Msun")
                    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(f))
                                    for f in frequency_hz[band]])
                    pairs = {name: weighted.optimized_overlap(
                        frequency_geom[band], spectra[candidate][band],
                        spectra["nr_high"][band], psd,
                        bounds=tuple(plan["allowed_time_shift_M"]),
                        step=plan["time_shift_grid_step_M"])
                        for name, candidate in (("native_vs_nr", "native"),
                                                ("low_vs_high_nr", "nr_low"))}
                    rows.append(dict(total_mass_msun=mass,
                                     lower_frequency_Hz=lower_hz,
                                     upper_frequency_Hz=plan["maximum_frequency_Hz"],
                                     frequency_bins=int(np.count_nonzero(band)),
                                     results=pairs))
                return dict(taper_width_M=width, common_time_support_M=support,
                            fft_length=fft_length, rows=rows)

            primary = sweep(plan["taper_width_M"], masses)
            controls = [sweep(width, plan["taper_control_masses_msun"])
                        for width in plan["taper_control_widths_M"]]
            values = [row["results"]["native_vs_nr"]["overlap"]
                      for row in primary["rows"]]
            control_values = [row["results"]["low_vs_high_nr"]["overlap"]
                              for row in primary["rows"]]
            minimum = min(values)
            result["cases"][simulation] = dict(
                array_sha256=q.sha256(array_path),
                nr_high_metadata_sha256=q.sha256(high_metadata_path),
                nr_reference_mode_Momega=2 * orbital,
                nr_relaxed_start_after_anchor_M=settled_start,
                primary=primary, taper_controls=controls,
                minimum_native_vs_nr=minimum,
                mass_at_minimum_msun=masses[values.index(minimum)],
                minimum_low_vs_high_nr=min(control_values),
                selected_case_99_percent_passed=minimum
                >= plan["selected_case_99_percent_threshold"],
                time_shift_at_boundary=any(
                    pair["time_shift_at_boundary"] for row in primary["rows"]
                    for pair in row["results"].values()), errors=[])
            print(f"{simulation} weighted holdout complete", flush=True)
        except Exception as exc:
            result["cases"][simulation] = dict(
                errors=[dict(type=type(exc).__name__, message=str(exc))])
            print(f"{simulation} weighted holdout failed: {exc}", flush=True)
        write(args.json_out, result)
    result["complete_case_set"] = set(result["cases"]) == set(case_plan["cases"])
    result["all_cases_succeeded"] = (result["complete_case_set"] and all(
        not item["errors"] for item in result["cases"].values()))
    result["all_selected_cases_99_percent_passed"] = (
        result["all_cases_succeeded"] and all(
            item["selected_case_99_percent_passed"]
            for item in result["cases"].values()))
    write(args.json_out, result)
    print(json.dumps(dict(all_cases_succeeded=result["all_cases_succeeded"],
                          all_selected_cases_99_percent_passed=
                          result["all_selected_cases_99_percent_passed"])), flush=True)
    return 0 if result["all_cases_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
