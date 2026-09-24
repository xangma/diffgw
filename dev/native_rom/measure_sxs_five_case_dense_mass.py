"""Refine the frozen five-case weighted proxy to a 1-solar-mass grid."""

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
    parser.add_argument("--dense-plan", type=Path, required=True)
    parser.add_argument("--weighted-plan", type=Path, required=True)
    parser.add_argument("--weighted-result", type=Path, required=True)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--only-simulation", help="Run one case for a smoke check")
    args = parser.parse_args()
    plan = json.loads(args.dense_plan.read_text())
    base_plan = json.loads(args.weighted_plan.read_text())
    baseline = json.loads(args.weighted_result.read_text())
    case_plan = json.loads(args.case_plan.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-five-case-dense-mass-plan.v1"
            or plan["qualification_claim"] is not False
            or q.sha256(args.weighted_result) != plan["weighted_result_sha256"]
            or q.sha256(args.weighted_plan) != plan["weighted_plan_sha256"]
            or q.sha256(weighted.__file__) != plan["weighted_function_source_sha256"]
            or baseline["plan_sha256"] != plan["weighted_plan_sha256"]
            or baseline["complete_case_set"] is not True
            or q.sha256(args.case_plan) != base_plan["case_plan_sha256"]
            or q.sha256(args.decoded_dir / "extraction.json")
            != base_plan["extraction_sha256"]
            or set(baseline["cases"]) != set(case_plan["cases"])
            or plan["mass_step_msun"] != 1
            or plan["mass_min_msun"] != 10
            or plan["mass_max_msun"] != 200):
        raise ValueError("dense-mass prerequisites changed")
    if args.only_simulation and args.only_simulation not in baseline["cases"]:
        raise ValueError("unknown simulation")
    import lalsimulation as lalsim
    from torchwave.constants import MSUN_SEC
    q.lal_binary_identity(base_plan["binary_versions"]["lal"],
                          base_plan["binary_versions"]["lalsimulation"])
    extraction = json.loads((args.decoded_dir / "extraction.json").read_text())
    pairs = {"native_vs_nr": "native", "public_vs_nr": "public",
             "low_vs_high_nr": "nr_low"}
    output = dict(schema="diffgw.seobnrv4.sxs-five-case-dense-mass.v1",
                  plan_sha256=q.sha256(args.dense_plan),
                  script_sha256=q.sha256(__file__), release_qualified=False,
                  complete_case_set=args.only_simulation is None, cases={})
    masses = range(plan["mass_min_msun"], plan["mass_max_msun"] + 1,
                   plan["mass_step_msun"])
    for simulation, old in baseline["cases"].items():
        if args.only_simulation and simulation != args.only_simulation:
            continue
        number = simulation.rsplit(":", 1)[-1]
        array_path = args.decoded_dir / f"sxs_{number}_long_start_arrays.npz"
        if q.sha256(array_path) != old["array_sha256"]:
            raise ValueError(f"long-start array changed: {simulation}")
        low = min(case_plan["cases"][simulation]["levels"])
        record = extraction["cases"][simulation]["extracted"][f"Lev{low}"]
        nr_low_path = args.decoded_dir / record["path"]
        if (q.sha256(nr_low_path) != old["nr_low_sha256"]
                or q.sha256(nr_low_path) != record["sha256"]):
            raise ValueError(f"lower NR mode changed: {simulation}")
        with np.load(array_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"])
                     for name in ("native", "public", "nr")}
        waves["nr_low"] = c.anchor(c.load_mode(nr_low_path),
                                   dt=base_plan["sample_step_M"])
        f_geom, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=old["nr_relaxed_start_after_anchor_M"],
            dt=base_plan["sample_step_M"], width=base_plan["taper_width_M"])
        if (support != old["primary"]["common_time_support_M"]
                or fft_length != old["primary"]["fft_length"]):
            raise ValueError(f"weighted support changed: {simulation}")
        orbital = old["nr_reference_mode_Momega"] / 2
        rows = []
        for mass in masses:
            mass_seconds = mass * float(MSUN_SEC)
            hz = f_geom / mass_seconds
            lower_hz = max(base_plan["minimum_frequency_Hz"],
                           orbital / (math.pi * mass_seconds))
            band = (hz >= lower_hz) & (hz <= base_plan["maximum_frequency_Hz"])
            frequency = f_geom[band]
            if len(frequency) < 8:
                raise ValueError(f"insufficient band: {simulation}, {mass} Msun")
            psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(x))
                            for x in hz[band]])
            results = {name: weighted.optimized_overlap(
                frequency, spectra[candidate][band], spectra["nr"][band], psd,
                bounds=tuple(base_plan["allowed_time_shift_M"]),
                step=base_plan["time_shift_grid_step_M"])
                for name, candidate in pairs.items()}
            rows.append(dict(total_mass_msun=mass, results=results))
        original = {row["total_mass_msun"]: row for row in old["primary"]["rows"]}
        for row in rows:
            old_row = original.get(row["total_mass_msun"])
            if old_row is None:
                continue
            for name in pairs:
                if abs(row["results"][name]["overlap"]
                       - old_row["results"][name]["overlap"]) > plan[
                           "baseline_overlap_absolute_tolerance"]:
                    raise ValueError(f"baseline overlap drift: {simulation}, {name}")
        summary = {}
        for name in pairs:
            minimum = min(row["results"][name]["overlap"] for row in rows)
            summary[name] = dict(
                minimum_overlap=minimum,
                minimum_masses_msun=[row["total_mass_msun"] for row in rows
                                     if row["results"][name]["overlap"] == minimum],
                time_shift_boundary_hits=sum(row["results"][name]
                                             ["time_shift_at_boundary"] for row in rows))
        output["cases"][simulation] = dict(
            array_sha256=q.sha256(array_path),
            nr_low_sha256=q.sha256(nr_low_path),
            masses_msun=list(masses),
            overlap={name: [row["results"][name]["overlap"] for row in rows]
                     for name in pairs},
            best_lag_M={name: [row["results"][name]["best_lag_M"] for row in rows]
                        for name in pairs},
            summary=summary)
        print(f"{simulation} dense mass sweep complete: "
              f"{summary['native_vs_nr']['minimum_overlap']:.8f}", flush=True)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, allow_nan=False,
                                             sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
