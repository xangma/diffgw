"""Localize the frozen aligned-spin post-peak mismatch in fixed windows."""

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
from .diagnose_late_mode_shape import _sample
from .diagnose_peak_anchor import fitted_peak, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/late_tail_shape_plan.json"


def tail_shape(native_time, native_mode, reference_time, reference_mode, mass_seconds, plan):
    npk = fitted_peak(native_time, native_mode, "quadratic_power")
    rpk = fitted_peak(reference_time, reference_mode, "quadratic_power")
    nt = (native_time - npk["time_s"]) / mass_seconds
    rt = (reference_time - rpk["time_s"]) / mass_seconds
    nm = native_mode * np.exp(-1j * npk["phase_rad"])
    rm = reference_mode * np.exp(-1j * rpk["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t, reference = rt[common], rm[common]
    candidate = _sample(nt, nm, t)
    if t[0] > 0 or t[-1] < max(plan["sample_times_M"]):
        raise ValueError("frozen late-tail samples exceed common waveform support")
    post = t >= 0
    post_norm = np.linalg.norm(reference[post])
    samples = []
    reference_peak_amplitude = abs(_sample(rt, rm, 0.))
    native_peak_amplitude = abs(_sample(nt, nm, 0.))
    for point in plan["sample_times_M"]:
        native = _sample(nt, nm, point)
        public = _sample(rt, rm, point)
        samples.append(dict(
            time_M=point,
            native_amplitude_to_peak=float(abs(native) / native_peak_amplitude),
            public_amplitude_to_peak=float(abs(public) / reference_peak_amplitude),
            native_to_public_amplitude=(float(abs(native) / abs(public)) if abs(public) > 0 else None),
            phase_residual_rad=(float(np.angle(native * public.conjugate()))
                                if abs(native) > 0 and abs(public) > 0 else None)))
    windows = []
    for low, high in plan["windows_M"]:
        mask = (t >= low) & (t < high)
        if mask.sum() < 3:
            raise ValueError("too few samples in frozen late-tail window")
        windows.append(dict(time_M=[low, high],
                            public_squared_norm_fraction=float(np.linalg.norm(reference[mask])**2 /
                                                               post_norm**2),
                            squared_error_fraction=float(np.linalg.norm(candidate[mask] - reference[mask])**2 /
                                                         post_norm**2),
                            metrics=q.mode_metrics(candidate[mask], reference[mask])))
    return dict(samples=samples, windows=windows,
                post_peak=q.mode_metrics(candidate[post], reference[post]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_shape"]
    prior = json.loads(prior_path.read_text())
    choices = json.loads((q.ROOT / plan["convergence_plan"]).read_text())
    if (plan["schema"] != "diffgw.seobnrv4.late-tail-shape-plan.v1"
            or q.sha256(prior_path) != plan["prior_shape_sha256"]
            or prior["status"] != "diagnostic_complete"
            or plan["row_index"] != 1 or plan["level_name"] != "ode_2"):
        parser.error("frozen tail plan or prior differs")
    torch.set_num_threads(1)
    level = next(item for item in choices["ode_levels"] if item["name"] == plan["level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    row = prior["row"]
    report = dict(schema="diffgw.seobnrv4.late-tail-shape.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256=q.sha256(prior_path), script_sha256=q.sha256(__file__),
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
        native = q.summarize_run(run)
        if native["array_sha256"] != prior["result"]["native"]["array_sha256"]:
            raise ValueError("native replay differs from frozen prior")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        shape = tail_shape(nt, nm, rt, rm, mass_seconds, plan)
        old = prior["result"]["shape"]["post_peak_baseline"]["complex_relative_l2"]
        if not np.isclose(shape["post_peak"]["complex_relative_l2"], old, rtol=0, atol=1e-12):
            raise ValueError("public replay differs from frozen post-peak comparison")
        report["result"] = dict(native=native, shape=shape,
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
