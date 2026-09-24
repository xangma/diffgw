"""Differentiate pinned-source NQC inputs along its own independently integrated field."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import solve_ivp

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import RESAMPLE_ANCHOR
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_dense_field_plan.json"
CALL_ANCHOR = "            if (XLALSimIMRSpinEOBCalculateNQCCoefficientsV4\n"
STATE_CAPTURE = """\
            if (modeL == 2 && modeM == 2 && diffgw_state_data
                && rHi.length <= diffgw_state_capacity
                && phiHi.length == rHi.length && pPhiHi.length == rHi.length) {
              diffgw_state_count = rHi.length;
              for (size_t diffgw_i = 0; diffgw_i < rHi.length; ++diffgw_i) {
                diffgw_state_data[2*diffgw_i] = phiHi.data[diffgw_i];
                diffgw_state_data[2*diffgw_i+1] = pPhiHi.data[diffgw_i];
              }
            }
"""
NAMES = ("amplitude", "A*q1", "A*q2", "A*q3", "mode_phase", "p1", "p1*pr2")


def source_series(amplitude_phase_r_pr_omega):
    amplitude, phase, radius, momentum, omega = amplitude_phase_r_pr_omega.T
    p1 = momentum/(radius*omega)
    q1 = p1*p1
    return np.column_stack((amplitude, amplitude*q1, amplitude*q1/radius,
                            amplitude*q1/radius**1.5, phase, p1,
                            p1*momentum*momentum))


def five_point(values, step):
    first = (values[0]-8*values[1]+8*values[3]-values[4])/(12*step)
    second = (-values[0]+16*values[1]-30*values[2]
              +16*values[3]-values[4])/(12*step*step)
    return dict(first=first.tolist(), second=second.tolist(),
                center=values[2].tolist())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--row-name", default="strong_aligned_spin")
    parser.add_argument("--resample-factor", type=int, default=32)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_high_rate", "source_high_rate_domain", "source_wheel_mode", "source_flux_plan",
             "capture_adapter", "field_adapter", "mode_adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-dense-field-plan.v1"
            or plan["row_names"] != ["strong_aligned_spin", "nonspinning", "mild_mixed_spin"]
            or args.row_name not in plan["row_names"]
            or plan["output_step_M"] != 0.125
            or plan["resample_factors"] != [8, 16, 32]
            or args.resample_factor not in plan["resample_factors"]
            or plan["capacity"] != 50000
            or plan["anchor_before_match_M"] != 4.0
            or plan["end_after_match_M"] != 0.5
            or plan["five_point_steps_M"] != [0.2, 0.1, 0.05, 0.025]
            or plan["dop853_tolerances"] != [[1e-11, 1e-12], [1e-12, 1e-13]]
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["source_high_rate"].read_text())
    domain_prior = json.loads(paths["source_high_rate_domain"].read_text())
    wheel = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or domain_prior["status"] != "diagnostic_complete"
            or wheel["status"] != "diagnostic_complete"):
        parser.error("Frozen source results unavailable")
    if args.row_name == "strong_aligned_spin":
        item = next(x for x in prior["rows"]
                    if x["resample_factor"] == args.resample_factor)
    else:
        item = next(x for x in domain_prior["rows"]
                    if x["name"] == args.row_name
                    and x["resample_factor"] == args.resample_factor)
    row = next(x["row"] for x in wheel["rows"]
               if x["name"] == args.row_name)
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-dense-field.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             independent_source_field=True),
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(
                      (CAPTURE+STATE_CAPTURE).encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  row_name=args.row_name, resample_factor=args.resample_factor,
                  row=row, source={},
                  component_consistency={}, dop853=[],
                  comparison={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-dense-field-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if (original.count(INCLUDE) != 1 or original.count(RESAMPLE_ANCHOR) != 1
                    or original.count(CALL_ANCHOR) != 1):
                raise ValueError("Pinned source diagnostic anchors changed")
            instrumented = original.replace(INCLUDE, INCLUDE+CAPTURE)
            instrumented = instrumented.replace(
                RESAMPLE_ANCHOR,
                f"\n\n  resampFac = {args.resample_factor}u; /* diagnostic high-rate grid */"
                + RESAMPLE_ANCHOR)
            instrumented = instrumented.replace(CALL_ANCHOR, STATE_CAPTURE+CALL_ANCHOR)
            source_file.write_text(instrumented)
            report["source"]["instrumented_source_sha256"] = q.sha256(source_file)
            capture_dylib = Path(tmp) / "capture.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["capture_adapter"], capture_dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(capture_dylib)).diffgw_v4_full_source_nqc_states
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            capacity = plan["capacity"]
            count = ctypes.c_size_t()
            meta = np.empty(8, dtype=np.float64)
            values = np.empty((capacity, 5), dtype=np.float64)
            coefficients = np.empty(10, dtype=np.float64)
            extra = np.empty((capacity, 2), dtype=np.float64)
            def pointer(array):
                return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            rc = function(*[row[key] for key in ("mass1", "mass2", "chi1", "chi2")],
                          row["omega_start"], plan["output_step_M"], capacity,
                          ctypes.byref(count), pointer(meta), pointer(values),
                          pointer(coefficients), pointer(extra))
            if rc or count.value < 10 or count.value > capacity:
                raise RuntimeError(f"Pinned source state capture failed: {rc}")
            values = values[:count.value].copy()
            extra = extra[:count.value].copy()
            active = coefficients[[0, 1, 2, 6, 7]]
            array_hash = hashlib.sha256(values.tobytes()).hexdigest()
            if (array_hash != item["input_array_sha256"]
                    or count.value != item["samples"]
                    or abs(meta[2]-item["match_time_M"]) > 1e-12
                    or np.max(abs(active-item["active_nqc_coefficients"])) > 1e-9):
                raise ValueError("Source capture fails frozen high-rate replay")
            report["source"].update(input_array_sha256=array_hash,
                                    state_extra_sha256=hashlib.sha256(
                                        extra.tobytes()).hexdigest(),
                                    samples=count.value, meta=meta.tolist(),
                                    active_nqc_coefficients=active.tolist())
            q.write_evidence(args.json_out, report)
            source_file.write_text(original)
            field_dylib = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["field_adapter"], field_dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            field_function = ctypes.CDLL(str(field_dylib)).diffgw_v4_source_field
            field_function.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
            field_function.restype = ctypes.c_int
            mode_dylib = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_dylib,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            mode_function = ctypes.CDLL(str(mode_dylib)).diffgw_v4_mode22_same_state
            mode_function.argtypes = [ctypes.c_double]*9 + [ctypes.POINTER(ctypes.c_double)]
            mode_function.restype = ctypes.c_int
            masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
            x1 = row["mass1"]/(row["mass1"]+row["mass2"])

            def source_field(state):
                out = (ctypes.c_double*6)()
                rc = field_function(*masses, *state, out)
                if rc:
                    raise RuntimeError(f"Source field failed: {rc}")
                return np.asarray(out[:4])

            def source_mode(state, omega):
                out = (ctypes.c_double*39)()
                rc = mode_function(x1, 1-x1, row["chi1"], row["chi2"],
                                   *state, omega, out)
                if rc:
                    raise RuntimeError(f"Source mode failed: {rc}")
                return complex(out[0], out[1])

            axis = np.arange(len(values), dtype=np.float64)*meta[1]
            match = float(meta[2])
            anchor_index = int(np.argmin(abs(axis-(match-plan["anchor_before_match_M"]))))
            anchor_state = np.array((values[anchor_index, 2], extra[anchor_index, 0],
                                     values[anchor_index, 3], extra[anchor_index, 1]))
            at_match = int(np.argmin(abs(axis-match)))
            match_state = np.array((values[at_match, 2], extra[at_match, 0],
                                    values[at_match, 3], extra[at_match, 1]))
            checks = {}
            for name, index, state in (("anchor", anchor_index, anchor_state),
                                       ("nearest_match", at_match, match_state)):
                omega = source_field(state)[1]
                mode = source_mode(state, values[index, 4])
                checks[name] = dict(index=index, time_M=float(axis[index]),
                    state=state.tolist(), omega_field=float(omega),
                    omega_captured=float(values[index, 4]),
                    amplitude_mode=abs(mode), amplitude_captured=float(values[index, 0]),
                    amplitude_relative_error=float(abs(mode)/values[index, 0]-1),
                    phase_wrapped_error=float(np.angle(np.exp(1j*(np.angle(mode)-values[index, 1])))))
            report["component_consistency"] = checks
            q.write_evidence(args.json_out, report)
            if (max(abs(x["amplitude_relative_error"]) for x in checks.values()) > 1e-6
                    or max(abs(x["phase_wrapped_error"]) for x in checks.values()) > 1e-6):
                raise ValueError("Explicit source mode does not reproduce captured pre-NQC mode")
            near = np.flatnonzero(abs(axis-match) <= 2.0)
            amplitude_errors = []
            phase_errors = []
            omega_errors = []
            for index in near:
                state = np.array((values[index, 2], extra[index, 0],
                                  values[index, 3], extra[index, 1]))
                mode = source_mode(state, values[index, 4])
                amplitude_errors.append(abs(abs(mode)-values[index, 0]))
                phase_errors.append(abs(np.angle(
                    np.exp(1j*(np.angle(mode)-values[index, 1])))))
                omega_errors.append(abs(source_field(state)[1]-values[index, 4]))
            report["component_consistency"]["local_window"] = dict(
                half_width_M=2.0, samples=len(near),
                max_amplitude_abs_error=float(max(amplitude_errors)),
                max_phase_wrapped_error=float(max(phase_errors)),
                max_omega_abs_error=float(max(omega_errors)))
            native_series = source_series(values)
            captured = np.asarray([gsl.derivatives(axis, native_series[:, j], match)
                                   for j in range(len(NAMES))])
            captured_state_derivatives = np.asarray([
                gsl.derivatives(axis, series, match)
                for series in (values[:, 2], extra[:, 0], values[:, 3], extra[:, 1])
            ])
            captured_state = captured_state_derivatives[:, 0]
            field_at_captured_state = source_field(captured_state)
            captured_mode = gsl.derivatives(axis, values[:, 0], match)[0]
            centered_omega = source_field(captured_state)[1]
            centered_mode_amplitude = abs(source_mode(captured_state, centered_omega))
            report["source"].update(match_time_M=match, anchor_time_M=float(axis[anchor_index]),
                                    anchor_state=anchor_state.tolist(),
                                    captured_state_at_match=captured_state.tolist(),
                                    captured_state_first_derivative=(
                                        captured_state_derivatives[:, 1]).tolist(),
                                    field_at_captured_state=field_at_captured_state.tolist(),
                                    captured_minus_field_first_derivative=(
                                        captured_state_derivatives[:, 1]-field_at_captured_state
                                    ).tolist(),
                                    captured_amplitude_at_match=captured_mode,
                                    explicit_mode_amplitude_at_match=centered_mode_amplitude,
                                    explicit_minus_captured_amplitude_at_match=(
                                        centered_mode_amplitude-captured_mode),
                                    captured_gsl_derivatives=captured.tolist())
            q.write_evidence(args.json_out, report)

            def scipy_field(_time, state):
                return source_field(state)

            def stencils_for(state_at):
                stencils = {}
                for step in plan["five_point_steps_M"]:
                    times = match+np.arange(-2, 3)*step
                    sample = []
                    for coordinate in times:
                        state = state_at(coordinate)
                        omega = source_field(state)[1]
                        mode = source_mode(state, omega)
                        radius, _phi, momentum, _pphi = state
                        amplitude = abs(mode)
                        p1 = momentum/(radius*omega)
                        q1 = p1*p1
                        sample.append((amplitude, amplitude*q1, amplitude*q1/radius,
                                       amplitude*q1/radius**1.5, np.angle(mode), p1,
                                       p1*momentum*momentum))
                    sample = np.asarray(sample)
                    sample[:, 4] = np.unwrap(sample[:, 4])
                    stencils[f"{step:g}"] = five_point(sample, step)
                return stencils

            for rtol, atol in plan["dop853_tolerances"]:
                started = time.monotonic()
                solution = solve_ivp(scipy_field,
                                     (axis[anchor_index], match+plan["end_after_match_M"]),
                                     anchor_state, method="DOP853", rtol=rtol, atol=atol,
                                     dense_output=True)
                if not solution.success:
                    raise RuntimeError(solution.message)
                stencils = stencils_for(solution.sol)
                center_state = solution.sol(match)
                center_omega = source_field(center_state)[1]
                center_amplitude = abs(source_mode(center_state, center_omega))
                report["dop853"].append(dict(rtol=rtol, atol=atol, nfev=solution.nfev,
                    seconds=time.monotonic()-started, center_state=center_state.tolist(),
                    center_minus_captured_state=(center_state-captured_state).tolist(),
                    center_amplitude=center_amplitude,
                    center_minus_captured_amplitude=center_amplitude-captured_mode,
                    stencils=stencils))
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(dop_rtol=rtol, nfev=solution.nfev,
                                      seconds=report["dop853"][-1]["seconds"])), flush=True)
            report["centered_dop853"] = []
            extent = 2*max(plan["five_point_steps_M"])
            for rtol, atol in plan["dop853_tolerances"]:
                started = time.monotonic()
                backward = solve_ivp(scipy_field, (match, match-extent),
                                     captured_state, method="DOP853", rtol=rtol,
                                     atol=atol, dense_output=True)
                forward = solve_ivp(scipy_field, (match, match+extent),
                                    captured_state, method="DOP853", rtol=rtol,
                                    atol=atol, dense_output=True)
                if not backward.success or not forward.success:
                    raise RuntimeError("Centered source integration failed")
                stencils = stencils_for(
                    lambda t: backward.sol(t) if t < match else forward.sol(t))
                report["centered_dop853"].append(dict(
                    rtol=rtol, atol=atol, nfev_backward=backward.nfev,
                    nfev_forward=forward.nfev, seconds=time.monotonic()-started,
                    stencils=stencils))
                q.write_evidence(args.json_out, report)
            reference = report["centered_dop853"][-1]["stencils"]["0.1"]
            report["comparison"]["captured_minus_dense_first"] = (
                captured[:, 1]-reference["first"]).tolist()
            report["comparison"]["captured_minus_dense_second"] = (
                captured[:, 2]-reference["second"]).tolist()
            report["comparison"]["tolerance_second_spread"] = (
                np.asarray(report["centered_dop853"][0]["stencils"]["0.1"]["second"])
                -reference["second"]).tolist()
            report["comparison"]["step_second_spread"] = (
                np.asarray(report["centered_dop853"][-1]["stencils"]["0.05"]["second"])
                -reference["second"]).tolist()
            report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
