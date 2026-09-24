"""Post-hoc public SEOBNRv4 attribution control for failed SXS:BBH:2569."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--weighted-result", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    case_plan = json.loads(args.case_plan.read_text())
    weighted_plan = json.loads(args.weighted_plan.read_text())
    comparison = json.loads(args.comparison.read_text())
    weighted_result = json.loads(args.weighted_result.read_text())
    control = json.loads(args.control_plan.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    simulation = "SXS:BBH:2569"
    if (case_plan["schema"] != "diffgw.seobnrv4.sxs-third-holdout-plan.v1"
            or weighted_plan["schema"] != "diffgw.seobnrv4.sxs-third-holdout-weighted-plan.v1"
            or control["schema"] != "diffgw.seobnrv4.sxs-2569-public-control-plan.v1"
            or control["qualification_claim"] is not False
            or control["case_plan_sha256"] != q.sha256(args.case_plan)
            or control["weighted_plan_sha256"] != q.sha256(args.weighted_plan)
            or control["comparison_sha256"] != q.sha256(args.comparison)
            or control["weighted_result_sha256"] != q.sha256(args.weighted_result)
            or weighted_result["all_cases_succeeded"] is not True
            or weighted_result["cases"][simulation]["selected_case_99_percent_passed"] is not False
            or comparison["all_cases_succeeded"] is not True
            or comparison["plan_sha256"] != q.sha256(args.case_plan)
            or comparison["extraction_sha256"] != q.sha256(extraction_path)
            or extraction["plan_sha256"] != q.sha256(args.case_plan)
            or control["simulation"] != simulation
            or control["masses_msun"] != [10, 22, 50, 100, 200]):
        raise ValueError("public control inputs do not match frozen holdout")
    binary = q.lal_binary_identity(control["binary_versions"]["lal"],
                                   control["binary_versions"]["lalsimulation"])
    case = case_plan["cases"][simulation]
    entry = comparison["cases"][simulation]
    array_path = Path(entry["anchored_array"]["path"])
    if q.sha256(array_path) != entry["anchored_array"]["sha256"]:
        raise ValueError("anchored holdout array changed")
    high = max(case["levels"])
    metadata_path = (args.archive_dir / simulation.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    metadata_record = extraction["cases"][simulation]["archive_files"][
        f"Lev{high}:metadata.json"]
    if q.sha256(metadata_path) != metadata_record["sha256"]:
        raise ValueError("NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    if metadata["reference_time"] != metadata["relaxation_time"]:
        raise ValueError("NR reference time no longer equals relaxation time")
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    if not math.isclose(orbital, float(np.linalg.norm(
            case["reference_orbital_frequency"])), abs_tol=1e-12):
        raise ValueError("NR reference frequency changed")
    dt = case_plan["method"]["grid_step_M"]
    waveform = public.public_mode(case["row"], dt, control["public_call"])
    anchored = c.anchor(waveform, dt=dt)
    with np.load(array_path, allow_pickle=False) as source:
        waves = {name: dict(time=source[f"{name}_time"],
                            mode=source[f"{name}_mode"])
                 for name in ("native", "nr_low", "nr_high")}
    waves["public"] = anchored
    nr_record = extraction["cases"][simulation]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / nr_record["path"]
    if q.sha256(nr_path) != nr_record["sha256"]:
        raise ValueError("NR mode changed")
    nr = c.anchor(c.load_mode(nr_path), dt=dt)
    windows = case_plan["method"]["windows_relative_to_highest_resolution_amplitude_peak_M"]
    common_support = entry["comparisons"]["native_vs_high"]["common_support_M"]
    fixed = dict(public_vs_nr=c.compare(anchored, nr, windows, dt,
                                        support=common_support),
                 native_vs_public=c.compare(waves["native"] | {
                     "peak_after_anchor_M": entry["anchor"]["native"]["peak_after_anchor_M"],
                     "peak_amplitude": entry["anchor"]["native"]["peak_amplitude"]},
                     anchored, windows, dt, support=common_support))
    settled_start = metadata["relaxation_time"] - entry["anchor"][f"Lev{high}"]["anchor_time_M"]
    f_geom, spectra, support, fft_length = weighted.tapered_spectra(
        waves, lower=settled_start, dt=dt, width=weighted_plan["taper_width_M"])
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC

    rows = []
    for mass in control["masses_msun"]:
        mass_seconds = mass * float(MSUN_SEC)
        frequency_hz = f_geom / mass_seconds
        lower_hz = max(weighted_plan["minimum_frequency_Hz"],
                       orbital / (math.pi * mass_seconds))
        band = ((frequency_hz >= lower_hz)
                & (frequency_hz <= weighted_plan["maximum_frequency_Hz"]))
        psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(value))
                        for value in frequency_hz[band]])
        results = {name: weighted.optimized_overlap(
            f_geom[band], spectra[candidate][band], spectra["nr_high"][band], psd,
            bounds=tuple(weighted_plan["allowed_time_shift_M"]),
            step=weighted_plan["time_shift_grid_step_M"])
            for name, candidate in (("native_vs_nr", "native"),
                                    ("public_vs_nr", "public"),
                                    ("low_vs_high_nr", "nr_low"))}
        rows.append(dict(total_mass_msun=mass, lower_frequency_Hz=lower_hz,
                         frequency_bins=int(np.count_nonzero(band)), results=results))
    result = dict(schema="diffgw.seobnrv4.sxs-2569-public-control.v1",
                  case_plan_sha256=q.sha256(args.case_plan),
                  weighted_plan_sha256=q.sha256(args.weighted_plan),
                  comparison_sha256=q.sha256(args.comparison),
                  weighted_result_sha256=q.sha256(args.weighted_result),
                  control_plan_sha256=q.sha256(args.control_plan),
                  extraction_sha256=q.sha256(extraction_path),
                  anchored_array_sha256=q.sha256(array_path),
                  nr_mode_sha256=q.sha256(nr_path),
                  nr_metadata_sha256=q.sha256(metadata_path),
                  script_sha256=q.sha256(__file__), lal_binary=binary,
                  public_call={key: waveform[key] for key in (
                      "f_lower_Hz", "delta_t_s", "distance_Mpc", "inclination_rad")},
                  anchors=dict(public={key: value for key, value in anchored.items()
                                       if key not in ("time", "mode")}),
                  fixed_crossing=fixed, common_time_support_M=support,
                  fft_length=fft_length, detector_noise_weighted=True,
                  dominant_mode_proxy=True, posthoc_selected_case=True,
                  release_qualified=False, rows=rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False,
                                        sort_keys=True) + "\n")
    print(json.dumps({mass["total_mass_msun"]:
                      {name: data["overlap"] for name, data in mass["results"].items()}
                      for mass in rows}))


if __name__ == "__main__":
    main()
