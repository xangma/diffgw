"""Measure complex comparison-grid interpolation error around the v4 peak."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.interpolate import make_interp_spline
import torch

from . import qualify_generator as q
from .diagnose_late_mode_shape import _sample
from .diagnose_peak_anchor import fitted_peak, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/interpolation_control_plan.json"


def compare_interpolants(native_time, native_mode, reference_time, reference_mode,
                         mass_seconds, plan):
    npk = fitted_peak(native_time, native_mode, "quadratic_power")
    rpk = fitted_peak(reference_time, reference_mode, "quadratic_power")
    nt = (native_time - npk["time_s"]) / mass_seconds
    rt = (reference_time - rpk["time_s"]) / mass_seconds
    nm = native_mode * np.exp(-1j * npk["phase_rad"])
    rm = reference_mode * np.exp(-1j * rpk["phase_rad"])
    common = (rt >= nt[0]) & (rt <= nt[-1])
    t, reference = rt[common], rm[common]
    if len(t) < 10:
        raise ValueError("too little common waveform support")
    candidates = {
        "linear_complex": _sample(nt, nm, t),
        "cubic_complex": make_interp_spline(nt, nm, k=3)(t),
        "quintic_complex": make_interp_spline(nt, nm, k=5)(t),
    }
    if tuple(candidates) != tuple(plan["interpolants"]):
        raise ValueError("interpolants differ from frozen plan")
    output = {}
    for name, candidate in candidates.items():
        windows = []
        for low, high in plan["windows_M"]:
            mask = (t >= low) & (t < high)
            if mask.sum() < 3:
                raise ValueError("too few samples in frozen interpolation window")
            windows.append(dict(time_M=[low, high],
                                metrics=q.mode_metrics(candidate[mask], reference[mask])))
        output[name] = dict(whole=q.mode_metrics(candidate, reference), windows=windows,
                            post_peak=q.mode_metrics(candidate[t >= 0], reference[t >= 0]))
    cubic_minus_quintic = q.mode_metrics(candidates["cubic_complex"],
                                         candidates["quintic_complex"])
    return dict(native_peak=npk, public_peak=rpk, common_samples=len(t),
                native_grid_step_M=float(np.median(np.diff(nt))),
                public_grid_step_M=float(np.median(np.diff(rt))),
                methods=output, cubic_minus_quintic=cubic_minus_quintic)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    shape_path, tail_path = (q.ROOT / plan[key] for key in ("prior_shape", "prior_tail"))
    shape, tail = (json.loads(path.read_text()) for path in (shape_path, tail_path))
    choices = json.loads((q.ROOT / plan["convergence_plan"]).read_text())
    if (plan["schema"] != "diffgw.seobnrv4.interpolation-control-plan.v1"
            or q.sha256(shape_path) != plan["prior_shape_sha256"]
            or q.sha256(tail_path) != plan["prior_tail_sha256"]
            or shape["status"] != "diagnostic_complete"
            or tail["status"] != "diagnostic_complete"
            or shape["row"] != tail["row"]
            or plan["row_index"] != 1 or plan["level_name"] != "ode_2"):
        parser.error("frozen interpolation plan or inputs differ")
    torch.set_num_threads(1)
    level = next(item for item in choices["ode_levels"] if item["name"] == plan["level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    row = shape["row"]
    report = dict(schema="diffgw.seobnrv4.interpolation-control.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256={"shape": q.sha256(shape_path), "tail": q.sha256(tail_path)},
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
        native = q.summarize_run(run)
        if native["array_sha256"] != shape["result"]["native"]["array_sha256"]:
            raise ValueError("native replay differs from frozen prior")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        comparison = compare_interpolants(nt, nm, rt, rm, mass_seconds, plan)
        old = shape["result"]["shape"]["post_peak_baseline"]["complex_relative_l2"]
        if not np.isclose(comparison["methods"]["linear_complex"]["post_peak"]
                          ["complex_relative_l2"], old, rtol=0, atol=1e-12):
            raise ValueError("linear baseline differs from frozen prior")
        report["result"] = dict(native=native, comparison=comparison,
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
