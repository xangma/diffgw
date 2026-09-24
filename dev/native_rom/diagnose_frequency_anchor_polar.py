"""Recheck frozen q4/q8 public comparisons without complex-sample attenuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import public_mode
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/frequency_anchor_polar_plan.json"


def polar_interpolate(query, time, mode):
    """Interpolate amplitude and continuous phase on the native sample grid."""
    return np.interp(query, time, np.abs(mode)) * np.exp(
        1j * np.interp(query, time, np.unwrap(np.angle(mode))))


def _polar_metrics(nt, nm, rt, rm, mass_seconds, baseline):
    na, ra = baseline["native_anchor"], baseline["reference_anchor"]
    ntime = (nt - na["time_s"]) / mass_seconds
    rtime = (rt - ra["time_s"]) / mass_seconds
    nmode = nm * np.exp(-1j * na["phase_rad"])
    rmode = rm * np.exp(-1j * ra["phase_rad"])
    common = (rtime >= ntime[0]) & (rtime <= ntime[-1])
    times = rtime[common]
    reference = rmode[common]
    candidate = polar_interpolate(times, ntime, nmode)
    if times.size < 3:
        raise ValueError("too few common frequency-anchored samples")
    relative_peak = times - ra["peak_lag_M"]
    windows = {}
    for name, mask in (("inspiral", relative_peak < -20),
                       ("late_inspiral", (relative_peak >= -20) & (relative_peak < 0)),
                       ("post_peak", relative_peak >= 0)):
        if mask.sum() >= 3:
            windows[name] = q.mode_metrics(candidate[mask], reference[mask])
    return dict(metrics=q.mode_metrics(candidate, reference), windows=windows,
                common_support_M=[float(times[0]), float(times[-1])])


def _source_rkf_run(row, level, settings, physical, flux, numerical):
    return q.native_run(
        row, level, settings,
        diagnostic_fine_rkf45_resampling=True,
        diagnostic_local_nqc_derivatives=True,
        diagnostic_matching_second_derivatives=True,
        diagnostic_nqc_phase_window_M=numerical["nqc_phase_window_M"],
        diagnostic_coarse_initial_step_M=numerical["coarse_initial_step_M"],
        diagnostic_warm_start_Momega=numerical["warm_start_Momega"],
        diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
        diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
        diagnostic_mode_33_higher_spin=physical["physical"]["mode_33_higher_spin"],
        diagnostic_mode_43_higher_spin=physical["mode_43_higher_spin"],
        diagnostic_spin_truncation=physical["spin_truncation"],
        diagnostic_hamiltonian_log_derivative=physical["physical"][
            "hamiltonian_log_derivative"],
        diagnostic_hamiltonian_calibration=physical["calibration"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / plan["screens"][name]["path"] for name in plan["rows"]}
    if (plan["schema"] != "diffgw.seobnrv4.frequency-anchor-polar-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != plan["screens"][name]["sha256"]
                   for name in plan["rows"])):
        parser.error("frozen plan or evidence changed")
    screens = {name: json.loads(paths[name].read_text()) for name in plan["rows"]}
    if any(screen["status"] != "diagnostic_numerical_gates_passed"
           or not screen["all_numerical_checks_passed"]
           or screen["selected_rows"] != [name]
           for name, screen in screens.items()):
        parser.error("source-backed RKF screens did not pass")
    if len({screen["source_manifest_sha256"] for screen in screens.values()}) != 1:
        parser.error("source-backed RKF screens use different source manifests")
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.frequency-anchor-polar.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: plan["screens"][name]["sha256"]
                                for name in plan["rows"]},
                  rows={}, errors=[], process=dict(
                      host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                      argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for name in plan["rows"]:
            screen = screens[name]
            row = screen["rows"][name]["inputs"]
            settings = q.Settings(**screen["settings"])
            level = dict(name="ode_2", atol=settings.atol_levels[2],
                         rtol=settings.rtol_levels[2], dt_geom=settings.dt_levels_M[0])
            started = time.monotonic()
            run = _source_rkf_run(row, level, settings, screen["physical_choice"],
                                  screen["physical_choice"]["flux"],
                                  screen["numerical_choice"])
            summary = q.summarize_run(run)
            expected = screen["rows"][name]["runs"][2]["array_sha256"]
            if summary["array_sha256"] != expected:
                raise ValueError(f"{name} native waveform differs from frozen candidate")
            result = dict(row=row, native_array_sha256=summary["array_sha256"],
                          native_elapsed_seconds=time.monotonic() - started)
            report["rows"][name] = result
            q.write_evidence(args.json_out, report)
            print(f"{name} native waveform replayed", flush=True)
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            result.update(linear=linear, polar=polar, public_samples=len(rt))
            q.write_evidence(args.json_out, report)
            print(f"{name} polar comparison complete", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
