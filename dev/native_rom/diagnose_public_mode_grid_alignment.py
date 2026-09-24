"""Bracket public pre-NQC mode alignment on three pinned output grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call, row_result as public_boundary_row
from .diagnose_public_pre_nqc_mode import public_raw_mode_at


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_mode_grid_alignment_plan.json"


def grid_result(prior, source, frozen_boundary, step, shifts, margin, gsl):
    boundary = public_boundary_row(prior, step, gsl)
    if step == frozen_boundary["output_step_M"] and (
            boundary["mode_sha256"] != frozen_boundary["mode_sha256"]
            or boundary["high_dynamics_sha256"] != frozen_boundary["high_dynamics_sha256"]
            or boundary["public_coefficients"] != frozen_boundary["public_coefficients"]):
        raise ValueError(f"{prior['name']} nominal public boundary does not reproduce")
    output, _ = public_call(prior["row"], step, 0)
    modes, _, high_dynamics = output
    high = np.asarray(high_dynamics.data, dtype=float).reshape(5, -1)
    if q.canonical_hash(high.tolist()) != boundary["high_dynamics_sha256"]:
        raise ValueError(f"{prior['name']} public dynamics do not reproduce")
    mode = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
    strain = np.asarray(mode.data.data, dtype=np.complex128)
    if q.canonical_hash(strain.view(np.float64).tolist()) != boundary["mode_sha256"]:
        raise ValueError(f"{prior['name']} public mode does not reproduce")
    time = high[0] - high[0, 0]
    omega_spline = CubicSpline(time, high[2], bc_type="natural").derivative()
    scale = ((prior["row"]["mass1"] + prior["row"]["mass2"])
             * lal.MRSUN_SI / (100e6 * lal.PC_SI))
    results = []
    for native in source["comparisons"]:
        frequency = native["orbital_frequency"]
        query = brentq(lambda t: omega_spline(t) - frequency,
                       time[0], boundary["orbital_peak_time_M"], xtol=1e-12)
        if query + margin >= min(boundary["inferred_match_time_M"],
                                 boundary["public_mode_amplitude_peak_time_M"]):
            raise ValueError(f"{prior['name']} frequency {frequency} is too near attachment")
        shifted = []
        for shift in shifts:
            start = boundary["high_mode_start_index"] + shift
            if start < 0 or start + len(time) > len(strain):
                raise ValueError("shifted mode slice is outside public output")
            high_mode = strain[start:start + len(time)] / scale
            public = public_raw_mode_at(time, high, high_mode,
                                        boundary["public_coefficients"], query, gsl)
            shifted.append(dict(
                mode_start_shift_samples=shift, public=public,
                raw_amplitude_relative=(native["native_raw_amplitude"]
                                        / public["raw_amplitude"] - 1),
                raw_amplitude_derivative_difference=(native["native_raw_amplitude_derivative"]
                                                     - public["raw_amplitude_derivative"]),
                raw_gravitational_wave_frequency_relative=(
                    native["native_raw_gravitational_wave_frequency"]
                    / public["raw_gravitational_wave_frequency"] - 1),
            ))
        results.append(dict(
            orbital_frequency=frequency,
            public_peak_relative_time_M=query - boundary["orbital_peak_time_M"],
            public_match_time_margin_M=boundary["inferred_match_time_M"] - query,
            public_mode_peak_margin_M=boundary["public_mode_amplitude_peak_time_M"] - query,
            public_orbital_frequency_crosscheck=(
                gsl.derivatives(time, high[2], query)[1] - frequency),
            native_raw_amplitude=native["native_raw_amplitude"],
            native_raw_amplitude_derivative=native["native_raw_amplitude_derivative"],
            native_raw_gravitational_wave_frequency=(
                native["native_raw_gravitational_wave_frequency"]),
            shifted=shifted,
        ))
    return dict(
        output_step_M=step, mode_sha256=boundary["mode_sha256"],
        high_dynamics_sha256=boundary["high_dynamics_sha256"],
        public_coefficients=boundary["public_coefficients"],
        cut_real=boundary["cut_real"], cut_index=boundary["cut_index"],
        nominal_mode_start_index=boundary["high_mode_start_index"],
        orbital_peak_time_M=boundary["orbital_peak_time_M"],
        comparisons=results,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-mode-grid-alignment-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["public_output_steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["mode_start_shifts_samples"] != [-1, 0, 1]
            or plan["pre_attachment_margin_M"] != 1.0):
        parser.error("frozen alignment plan differs")
    source_files = ("coefficient_evidence", "boundary_evidence", "pre_nqc_evidence")
    paths = {key: q.ROOT / plan[key] for key in source_files}
    for key, path in paths.items():
        if q.sha256(path) != plan[key + "_sha256"]:
            parser.error(f"{key} differs")
    coefficients = json.loads(paths["coefficient_evidence"].read_text())
    boundary = json.loads(paths["boundary_evidence"].read_text())
    source = json.loads(paths["pre_nqc_evidence"].read_text())
    binary = coefficients["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary
            or boundary["lal_binary"] != binary or source["lal_binary"] != binary
            or any([row["name"] for row in report["rows"]] != plan["rows"]
                   for report in (coefficients, boundary, source))
            or q.sha256(q.ROOT / "dev/native_rom/diagnose_public_pre_nqc_mode.py")
            != source["script_sha256"]):
        parser.error("pinned public binary, rows, or source diagnostic differ")
    gsl_paths = [Path(path) for path in binary["binaries"] if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    report = dict(
        schema="diffgw.seobnrv4.public-mode-grid-alignment.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        gsl = GSLCubic(gsl_paths[0])
        for prior, bound, native in zip(coefficients["rows"], boundary["rows"], source["rows"]):
            grids = []
            for step in plan["public_output_steps_M"]:
                grids.append(grid_result(prior, native, bound, step,
                                         plan["mode_start_shifts_samples"],
                                         plan["pre_attachment_margin_M"], gsl))
                print(f"row={prior['name']} step={step}", flush=True)
            report["rows"].append(dict(name=prior["name"], row=prior["row"], grids=grids))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
