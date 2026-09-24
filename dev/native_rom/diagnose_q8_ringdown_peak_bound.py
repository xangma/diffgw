"""Development-only q8 ringdown curvature counterfactual from the published ansatz."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode
from .diagnose_q8_frequency_anchor import frequency_anchor
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_ringdown_peak_bound_plan.json"


def _frequency_metrics(nt, nm, rt, rm, mass_seconds, target):
    npk = fitted_peak(nt, nm, "quadratic_power")
    rpk = fitted_peak(rt, rm, "quadratic_power")
    na = frequency_anchor(nt, nm, mass_seconds, target, npk["time_s"])
    ra = frequency_anchor(rt, rm, mass_seconds, target, rpk["time_s"])
    ntime = (nt-na["time_s"])/mass_seconds
    rtime = (rt-ra["time_s"])/mass_seconds
    nmode = nm*np.exp(-1j*na["phase_rad"])
    rmode = rm*np.exp(-1j*ra["phase_rad"])
    common = (rtime >= ntime[0]) & (rtime <= ntime[-1])
    times = rtime[common]
    reference = rmode[common]
    candidate = (np.interp(times, ntime, nmode.real)
                 + 1j*np.interp(times, ntime, nmode.imag))
    if times.size < 3:
        raise ValueError("too few common frequency-anchored samples")
    peak_relative = times-ra["peak_lag_M"]
    windows = {}
    for name, mask in (("inspiral", peak_relative < -20),
                       ("late_inspiral", (peak_relative >= -20) & (peak_relative < 0)),
                       ("post_peak", peak_relative >= 0)):
        if mask.sum() >= 3:
            windows[name] = q.mode_metrics(candidate[mask], reference[mask])
    return dict(native_anchor=na, reference_anchor=ra,
                native_minus_reference_peak_lag_M=na["peak_lag_M"]-ra["peak_lag_M"],
                metrics=q.mode_metrics(candidate, reference), windows=windows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    anchor_path = q.ROOT / plan["frequency_anchor"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-ringdown-peak-bound-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["level_name"] != "ode_2"
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(anchor_path) != plan["frequency_anchor_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    baseline = json.loads(anchor_path.read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or baseline["row"] != row or baseline["choice"] != choice
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("campaign or pinned public binary differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_peak_anchor.py",
        "dev/native_rom/diagnose_q8_frequency_anchor.py",
        "src/torchwave/approximants/eob/merger_ringdown.py",
        "src/torchwave/approximants/eob/numerical_attachment.py",
    )
    report = dict(schema="diffgw.seobnrv4.q8-ringdown-peak-bound.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  campaign_sha256=q.sha256(campaign_path),
                  frequency_anchor_sha256=q.sha256(anchor_path),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  lal_binary=pin, row=row, choice=choice, ringdown_calls=[], errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        import torchwave.approximants.eob.numerical_attachment as attachment

        original = attachment.merger_ringdown_22

        def bounded_ringdown(t, sigma, amplitude, derivative, frequency, phase, shape):
            c1, c2 = shape.amplitude_rate, shape.amplitude_offset
            gamma = -sigma.imag
            tanh_c2 = torch.tanh(c2)
            needs_bound = (tanh_c2 < 0) & (gamma + 2*c1*tanh_c2 < 0)
            safe_denom = torch.where(tanh_c2 < 0, -2*tanh_c2,
                                     torch.ones_like(tanh_c2))
            limit = gamma/safe_denom
            bounded_c1 = torch.where(needs_bound, limit, c1)
            if bool((derivative != 0).any()):
                raise ValueError("peak-bound diagnostic requires zero matching amplitude slope")
            report["ringdown_calls"].append(dict(
                damping=float(gamma), original_c1=float(c1), c2=float(c2),
                bounded_c1=float(bounded_c1), changed=bool(needs_bound),
                original_curvature_over_amplitude=float(-gamma*(gamma+2*c1*tanh_c2)),
                bounded_curvature_over_amplitude=float(
                    -gamma*(gamma+2*bounded_c1*tanh_c2))))
            return original(t, sigma, amplitude, derivative, frequency, phase,
                            shape._replace(amplitude_rate=bounded_c1))

        started = time.monotonic()
        with patch.object(attachment, "merger_ringdown_22", bounded_ringdown):
            run = _replay(row, item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), choice,
                          item["fixed_restart_time_M"])
        summary = q.summarize_run(run)
        if (summary["array_sha256"]["time"]
                != item["runs"]["ode_2"]["native"]["array_sha256"]["time"]):
            raise ValueError("counterfactual changed the output time grid")
        if not report["ringdown_calls"] or not all(
                call["changed"] for call in report["ringdown_calls"]):
            raise ValueError("expected the peak bound to activate on every ringdown call")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started,
                                match_time_M=run["match_time_M"],
                                boundary_time_M=run["boundary_time_M"])
        q.write_evidence(args.json_out, report)
        print("bounded_native_waveform_complete", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        comparison = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
        if abs(comparison["reference_anchor"]["time_s"]
               - baseline["frequency_anchored"]["reference_anchor"]["time_s"]) > 1e-12:
            raise ValueError("pinned public frequency anchor changed")
        report["comparison"] = comparison
        report["peak_fitted"] = fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                                "quadratic_power")
        report["baseline"] = dict(
            frequency_anchored=baseline["frequency_anchored"]["metrics"],
            peak_lag_M=baseline["frequency_anchored"]["native_minus_reference_peak_lag_M"],
            peak_fitted=baseline["peak_fitted"]["metrics"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
