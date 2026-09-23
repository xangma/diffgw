"""Freeze aligned-spin amplitude and phase residuals near the SEOBNRv4 peak."""

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
from .diagnose_peak_anchor import fitted_peak, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/late_mode_shape_plan.json"


def _sample(time, mode, query):
    return np.interp(query, time, mode.real) + 1j * np.interp(query, time, mode.imag)


def mode_shape(native_time, native_mode, reference_time, reference_mode, mass_seconds, plan):
    """Return predeclared peak-relative samples and a peak-scale diagnostic ablation."""
    native_peak = fitted_peak(native_time, native_mode, "quadratic_power")
    reference_peak = fitted_peak(reference_time, reference_mode, "quadratic_power")
    nt = (native_time - native_peak["time_s"]) / mass_seconds
    rt = (reference_time - reference_peak["time_s"]) / mass_seconds
    nm = native_mode * np.exp(-1j * native_peak["phase_rad"])
    rm = reference_mode * np.exp(-1j * reference_peak["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t, reference = rt[common], rm[common]
    candidate = _sample(nt, nm, t)
    if len(t) < 5 or t[0] > min(plan["sample_times_M"]) or t[-1] < max(plan["sample_times_M"]):
        raise ValueError("frozen peak-relative samples exceed common waveform support")
    native_at_peak = abs(_sample(nt, nm, 0.))
    reference_at_peak = abs(_sample(rt, rm, 0.))
    scale = float(reference_at_peak / native_at_peak)
    post = t >= 0
    if post.sum() < 5:
        raise ValueError("too few post-peak samples")
    sample_times = np.asarray(plan["sample_times_M"], dtype=float)
    ns = _sample(nt, nm, sample_times)
    rs = _sample(rt, rm, sample_times)
    samples = [dict(time_M=float(s), native_to_public_amplitude=float(abs(a) / abs(b)),
                    phase_residual_rad=float(np.angle(a * b.conjugate())))
               for s, a, b in zip(sample_times, ns, rs)]
    low, high = plan["fit_window_M"]
    fit_times = np.linspace(low, high, 181)
    nfit = _sample(nt, nm, fit_times)
    rfit = _sample(rt, rm, fit_times)
    ratio = nfit / rfit
    log_amplitude_slope = np.polyfit(fit_times, np.log(abs(ratio)), 1)[0]
    phase_slope = np.polyfit(fit_times, np.unwrap(np.angle(ratio)), 1)[0]
    return dict(native_peak=native_peak, public_peak=reference_peak,
                samples=samples, peak_amplitude_scale=scale,
                post_peak_baseline=q.mode_metrics(candidate[post], reference[post]),
                post_peak_peak_scaled=q.mode_metrics(scale * candidate[post], reference[post]),
                fit_window_M=[low, high],
                log_amplitude_ratio_slope_per_M=float(log_amplitude_slope),
                phase_residual_slope_rad_per_M=float(phase_slope))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_physical"]
    public_path = q.ROOT / plan["public_waveform"]
    prior = json.loads(prior_path.read_text())
    public = json.loads(public_path.read_text())
    choices = json.loads((q.ROOT / plan["convergence_plan"]).read_text())
    if (plan["schema"] != "diffgw.seobnrv4.late-mode-shape-plan.v1"
            or q.sha256(prior_path) != plan["prior_physical_sha256"]
            or q.sha256(public_path) != plan["public_waveform_sha256"]
            or prior["status"] != "diagnostic_complete"
            or public["status"] != "diagnostic_complete"
            or plan["row_index"] != 1 or plan["level_name"] != "ode_2"
            or public["rows"] != [entry["row"] for entry in prior["results"]]
            or q.lal_binary_identity(public["lal_binary"]["lal_version"],
                                     public["lal_binary"]["lalsimulation_version"])
            != public["lal_binary"]):
        parser.error("frozen late-shape plan or reference differs")
    torch.set_num_threads(1)
    level = next(item for item in choices["ode_levels"] if item["name"] == plan["level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    row = prior["results"][plan["row_index"]]["row"]
    report = dict(schema="diffgw.seobnrv4.late-mode-shape.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256=q.sha256(prior_path), public_sha256=q.sha256(public_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "src/torchwave/approximants/eob/merger_ringdown.py",
                      "dev/native_rom/qualify_generator.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv), row=row, result=None, errors=[])
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        run = q.native_run(
            row, level, settings,
            diagnostic_hermite_resampling=choices["hermite_resampling"],
            diagnostic_local_nqc_derivatives=choices["local_nqc_phase_derivatives"],
            diagnostic_restart_time_M=choices["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=choices["coarse_first_step_M"],
            diagnostic_warm_start_Momega=choices["native_start_Momega"])
        frozen = prior["results"][plan["row_index"]]["comparisons"]["baseline"]["waveform"]
        native = q.summarize_run(run)
        if native["array_sha256"] != frozen["array_sha256"]:
            raise ValueError("native replay differs from frozen physical comparison")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        report["result"] = dict(native=native,
                                shape=mode_shape(nt, nm, rt, rm, mass_seconds, plan),
                                elapsed_seconds=time.monotonic() - started)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
