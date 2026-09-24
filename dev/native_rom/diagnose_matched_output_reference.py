"""Compare the public high dynamics using the frozen 0.125 M waveform output step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/matched_output_reference_plan.json"
SCOPE = {
    **q.SCOPE,
    "description": "Pinned public NQC and high-dynamics comparisons; no training or promotion.",
    "known_limitations": [
        limitation for limitation in q.SCOPE["known_limitations"]
        if not limitation.startswith("No time, phase, amplitude, or coefficient fitting")
    ] + [
        "The public solver computes NQC coefficients internally; no values are fitted to "
        "the native result by this diagnostic.",
        "Equal-frequency dynamics comparisons do not expose the public raw-mode "
        "matching inputs or establish whole-waveform fidelity.",
    ],
}


def row_result(prior, different, grid, plan, gsl):
    output, coefficients = public_call(prior["row"], plan["public_output_step_M"], 1)
    if (list(output) != [None, None, None]
            or not np.array_equal(coefficients[[0, 1, 2, 6, 7]],
                                  np.asarray(prior["public_coefficients"]))):
        raise ValueError(f"{prior['name']} frozen public coefficients do not reproduce")
    output, _ = public_call(prior["row"], plan["public_output_step_M"], 0)
    high = np.asarray(output[2].data, dtype=float).reshape(5, -1)
    time = high[0] - high[0, 0]
    step = plan["expected_public_high_step_M"]
    if (not np.isfinite(high).all()
            or not np.allclose(np.diff(time), step, rtol=0, atol=1e-9)):
        raise ValueError(f"{prior['name']} public high dynamics grid differs")
    approximate_omega = np.gradient(high[2], step, edge_order=2)
    peak_index = int(np.argmax(approximate_omega))
    if peak_index < 2 or peak_index >= len(time) - 2:
        raise ValueError(f"{prior['name']} orbital-frequency peak is at boundary")
    comparisons = []
    for frequency, old in zip(plan["fixed_orbital_frequencies"],
                              different["equal_frequency_comparisons"]):
        if old["orbital_frequency"] != frequency:
            raise ValueError(f"{prior['name']} frozen frequencies differ")
        query = brentq(lambda t: gsl.derivatives(time, high[2], t)[1] - frequency,
                       time[0], time[peak_index], xtol=1e-12)
        public = [gsl.derivatives(time, values, query)[0]
                  for values in (high[1], high[3])]
        native = np.asarray(old["native_r_pr_star"])
        other = np.asarray(old["public_r_pr_star"])
        comparisons.append(dict(
            orbital_frequency=frequency, public_high_time_M=query,
            public_frequency_root_residual=(
                gsl.derivatives(time, high[2], query)[1] - frequency),
            native_r_pr_star=native.tolist(),
            matched_output_public_r_pr_star=public,
            matched_output_native_relative=(native / public - 1).tolist(),
            different_output_public_r_pr_star=other.tolist(),
            public_output_step_effect_relative=(np.asarray(public) / other - 1).tolist(),
        ))
    fine_step = plan["expected_public_high_step_M"]
    fine_output, _ = public_call(prior["row"], fine_step, 0)
    fine_high = np.asarray(fine_output[2].data, dtype=float).reshape(5, -1)
    if q.canonical_hash(fine_high.tolist()) != different["public_high_dynamics_sha256"]:
        raise ValueError(f"{prior['name']} different-output high dynamics do not reproduce")
    fine_time = fine_high[0] - fine_high[0, 0]
    fine_omega = np.gradient(fine_high[2], fine_step, edge_order=2)
    fine_peak_index = int(np.argmax(fine_omega))
    near_peak = []
    for frequency in plan["near_peak_public_frequencies_by_row"][prior["name"]]:
        states = []
        for dynamics, times, index in ((high, time, peak_index),
                                        (fine_high, fine_time, fine_peak_index)):
            query = brentq(
                lambda t: gsl.derivatives(times, dynamics[2], t)[1] - frequency,
                times[0], times[index], xtol=1e-12)
            states.append(dict(
                high_time_M=float(dynamics[0, 0] + query),
                r_pr_star_pphi=[gsl.derivatives(times, dynamics[j], query)[0]
                                for j in (1, 3, 4)],
                frequency_root_residual=(
                    gsl.derivatives(times, dynamics[2], query)[1] - frequency),
            ))
        near_peak.append(dict(
            orbital_frequency=frequency,
            matched_output=states[0], finer_output=states[1],
            matched_over_finer_state_relative=(
                np.asarray(states[0]["r_pr_star_pphi"])
                / states[1]["r_pr_star_pphi"] - 1).tolist(),
        ))
    baseline = coefficients[[0, 1, 2, 6, 7]]
    output_grid_coefficients = []
    for point in grid["grids"]:
        other = np.asarray(point["public_coefficients"])
        output_grid_coefficients.append(dict(
            requested_output_step_M=point["output_step_M"],
            public_coefficients=other.tolist(),
            relative_to_matched_output=(other / baseline - 1).tolist(),
        ))
    return dict(
        name=prior["name"], row=prior["row"],
        matched_public_coefficients=baseline.tolist(),
        different_output_grid_coefficients=output_grid_coefficients,
        public_high_dynamics_sha256=q.canonical_hash(high.tolist()),
        public_high_dynamics_length=high.shape[1],
        public_high_dynamics_start_M=float(high[0, 0]),
        public_orbital_peak_sample_time_M=float(time[peak_index]),
        finer_output_orbital_peak_sample_time_M=float(fine_time[fine_peak_index]),
        near_peak_public_comparisons=near_peak,
        comparisons=comparisons,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.matched-output-reference-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["public_output_step_M"] != 0.125
            or plan["expected_public_high_step_M"] != 0.015625
            or plan["fixed_orbital_frequencies"]
            != [0.07, 0.08, 0.10, 0.12, 0.14, 0.16]
            or plan["near_peak_public_frequencies_by_row"] != {
                "strong_aligned_spin": [0.165, 0.168, 0.169],
                "strong_antialigned_spin": [0.165, 0.170, 0.172]}):
        parser.error("frozen matched-output plan differs")
    paths = {key: q.ROOT / plan[key] for key in
             ("coefficient_evidence", "different_output_evidence",
              "different_output_grid_evidence")}
    for key, path in paths.items():
        if q.sha256(path) != plan[key + "_sha256"]:
            parser.error(f"{key} differs")
    coefficients = json.loads(paths["coefficient_evidence"].read_text())
    different = json.loads(paths["different_output_evidence"].read_text())
    grids = json.loads(paths["different_output_grid_evidence"].read_text())
    binary = coefficients["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary
            or different["lal_binary"] != binary or grids["lal_binary"] != binary
            or [row["name"] for row in coefficients["rows"]] != plan["rows"]
            or [row["name"] for row in different["rows"]] != plan["rows"]
            or [row["name"] for row in grids["rows"]] != plan["rows"]
            or any(row["output_step_M"] != plan["public_output_step_M"]
                   for row in coefficients["rows"])):
        parser.error("pinned public binary, rows, or output step differ")
    gsl_paths = [Path(path) for path in binary["binaries"]
                 if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    report = dict(
        schema="diffgw.seobnrv4.matched-output-reference.v1", scope=SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        gsl = GSLCubic(gsl_paths[0])
        for prior, old, grid in zip(coefficients["rows"], different["rows"], grids["rows"]):
            report["rows"].append(row_result(prior, old, grid, plan, gsl))
            q.write_evidence(args.json_out, report)
            print(f"row={prior['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
