"""Five-case post-hoc detector-weighted h22 diagnostic; no release claim."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as c
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--four-long-plan", type=Path, required=True)
    parser.add_argument("--q5-long-plan", type=Path, required=True)
    parser.add_argument("--q5-weighted-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--long-dir", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--only-simulation", help="Run one case as an unpublished smoke check")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.weighted_plan.read_text())
    case_plan = json.loads(args.case_plan.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    if (plan["schema"] != "diffgw.seobnrv4.sxs-five-case-weighted-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["extraction_sha256"] != q.sha256(extraction_path)
            or plan["four_long_start_plan_sha256"] != q.sha256(args.four_long_plan)
            or plan["sxs_2508_long_start_plan_sha256"] != q.sha256(args.q5_long_plan)
            or plan["sxs_2508_weighted_plan_sha256"] != q.sha256(args.q5_weighted_plan)
            or plan["weighted_function_source_sha256"] != q.sha256(weighted.__file__)
            or set(plan["nr_high_metadata_sha256"]) != set(case_plan["cases"])):
        raise ValueError("weighted five-case prerequisites changed")
    extraction = json.loads(extraction_path.read_text())
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC
    binary = q.lal_binary_identity(plan["binary_versions"]["lal"],
                                   plan["binary_versions"]["lalsimulation"])
    if args.only_simulation and args.only_simulation not in case_plan["cases"]:
        raise ValueError("unknown smoke-check simulation")
    output = dict(schema="diffgw.seobnrv4.sxs-five-case-weighted.v1",
                  plan_sha256=q.sha256(args.weighted_plan),
                  script_sha256=q.sha256(__file__), lal_binary=binary,
                  detector_noise_weighted=True, dominant_mode_proxy=True,
                  posthoc_selected_cases=True, release_qualified=False,
                  complete_case_set=args.only_simulation is None,
                  cases={})
    for simulation, case in case_plan["cases"].items():
        if args.only_simulation and simulation != args.only_simulation:
            continue
        number = simulation.rsplit(":", 1)[-1]
        long_path = args.long_dir / f"sxs_{number}_long_start.json"
        long_result = json.loads(long_path.read_text())
        long_plan_hash = (plan["sxs_2508_long_start_plan_sha256"] if number == "2508"
                          else plan["four_long_start_plan_sha256"])
        array_path = args.decoded_dir / f"sxs_{number}_long_start_arrays.npz"
        high, low = max(case["levels"]), min(case["levels"])
        metadata_path = (args.archive_dir / simulation.replace(":", "_")
                         / f"Lev{high}" / "metadata.json")
        metadata = json.loads(metadata_path.read_text())
        orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
        if (long_result["long_plan_sha256"] != long_plan_hash
                or long_result["plan_sha256"] != q.sha256(args.case_plan)
                or long_result["array_sha256"] != q.sha256(array_path)
                or not long_result["sources_unchanged"]
                or (number == "2508" and q.sha256(long_path)
                    != plan["sxs_2508_long_start_result_sha256"])
                or (number != "2508" and long_result["simulation"] != simulation)
                or q.sha256(metadata_path)
                    != plan["nr_high_metadata_sha256"][simulation]
                or metadata["reference_time"] != metadata["relaxation_time"]
                or not math.isclose(orbital,
                    float(np.linalg.norm(case["reference_orbital_frequency"])),
                    abs_tol=1e-12)):
            raise ValueError(f"settled waveform provenance changed: {simulation}")
        record = extraction["cases"][simulation]["extracted"][f"Lev{low}"]
        nr_low_path = args.decoded_dir / record["path"]
        if q.sha256(nr_low_path) != record["sha256"]:
            raise ValueError(f"NR lower resolution changed: {simulation}")
        with np.load(array_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"])
                     for name in ("native", "public", "nr")}
        waves["nr_low"] = c.anchor(c.load_mode(nr_low_path),
                                   dt=plan["sample_step_M"])
        relaxed_start = (metadata["relaxation_time"]
                         - long_result["anchors"]["nr"]["anchor_time_M"])
        late_starts = {name: float(wave["time"][0]) for name, wave in waves.items()
                       if float(wave["time"][0]) > relaxed_start + plan["sample_step_M"]}
        if late_starts:
            raise ValueError(f"waveform begins after settled NR reference: {simulation}, "
                             f"{late_starts}")
        def sweep(width, selected_masses):
            f_geom, spectra, support, fft_length = weighted.tapered_spectra(
                waves, lower=relaxed_start,
                dt=plan["sample_step_M"], width=width)
            rows = []
            for mass in selected_masses:
                mass_seconds = mass * float(MSUN_SEC)
                hz = f_geom / mass_seconds
                lower_hz = max(plan["minimum_frequency_Hz"],
                               orbital / (math.pi * mass_seconds))
                band = (hz >= lower_hz) & (hz <= plan["maximum_frequency_Hz"])
                frequency = f_geom[band]
                if len(frequency) < 8:
                    raise ValueError(f"insufficient band: {simulation}, {mass} Msun")
                psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(x))
                                for x in hz[band]])
                results = {
                    name: weighted.optimized_overlap(
                        frequency, spectra[candidate][band], spectra["nr"][band], psd,
                        bounds=tuple(plan["allowed_time_shift_M"]),
                        step=plan["time_shift_grid_step_M"])
                    for name, candidate in (("native_vs_nr", "native"),
                                            ("public_vs_nr", "public"),
                                            ("low_vs_high_nr", "nr_low"))}
                rows.append(dict(total_mass_msun=mass, lower_frequency_Hz=lower_hz,
                                 upper_frequency_Hz=plan["maximum_frequency_Hz"],
                                 frequency_bins=len(frequency), results=results))
            return dict(taper_width_M=width, common_time_support_M=support,
                        fft_length=fft_length, rows=rows)
        primary = sweep(plan["taper_width_M"], plan["total_masses_msun"])
        controls = [sweep(width, plan["taper_control_masses_msun"])
                    for width in plan["taper_control_widths_M"]]
        output["cases"][simulation] = dict(
            long_result_sha256=q.sha256(long_path), array_sha256=q.sha256(array_path),
            nr_high_metadata_sha256=q.sha256(metadata_path),
            nr_low_sha256=q.sha256(nr_low_path),
            nr_reference_mode_Momega=2 * orbital,
            nr_relaxed_start_after_anchor_M=relaxed_start,
            primary=primary, taper_controls=controls)
        print(f"{simulation} weighted sweep complete", flush=True)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, allow_nan=False,
                                             sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
