"""Profile untouched q4 postpeak residual under the frozen frequency anchor."""

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
from .diagnose_q8_frequency_anchor import frequency_anchor
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_postpeak_profile_plan.json"


def _aligned_series(time, mode, mass_seconds, anchor, reference_peak_lag):
    relative_time_M = (time-anchor["time_s"])/mass_seconds-reference_peak_lag
    aligned_mode = mode*np.exp(-1j*anchor["phase_rad"])
    return relative_time_M, aligned_mode


def _interp_complex(x, y, target):
    if target < x[0] or target > x[-1]:
        raise ValueError("profile target outside waveform support")
    return np.interp(target, x, y.real)+1j*np.interp(target, x, y.imag)


def _slope(x, y, lo, hi):
    selected = (x >= lo) & (x <= hi)
    if selected.sum() < 20:
        raise ValueError("too few slope samples")
    t, h = x[selected], y[selected]
    return dict(samples=int(t.size),
                log_amplitude_slope_per_M=float(np.polyfit(t, np.log(abs(h)), 1)[0]),
                phase_slope_rad_per_M=float(np.polyfit(t, np.unwrap(np.angle(h)), 1)[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    anchor_path = q.ROOT / plan["frequency_anchor"]
    if (plan["schema"] != "diffgw.seobnrv4.q4-postpeak-profile-plan.v1"
            or plan["row_name"] != "q4_mixed_spin"
            or plan["level_name"] != "ode_2"
            or plan["relative_times_M"] != [0, 5, 10, 20, 40, 80, 120]
            or plan["slope_windows_M"] != [[30, 70], [70, 120]]
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(anchor_path) != plan["frequency_anchor_sha256"]):
        parser.error("frozen plan or anchor differs")
    baseline = json.loads(anchor_path.read_text())
    campaign = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-24/source_candidate_new_rows.json").read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    if (baseline["status"] != "diagnostic_complete"
            or baseline["row"] != row or baseline["choice"] != choice):
        parser.error("frozen baseline differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_q8_frequency_anchor.py",
        "src/torchwave/approximants/eob/numerical_attachment.py",
    )
    report = dict(schema="diffgw.seobnrv4.q4-postpeak-profile.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  frequency_anchor_sha256=q.sha256(anchor_path),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  row=row, choice=choice, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        run = _replay(row, item["runs"]["ode_2"]["level"],
                      q.Settings(**campaign["settings"]), choice,
                      item["fixed_restart_time_M"])
        summary = q.summarize_run(run)
        if summary["array_sha256"] != item["runs"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("native waveform differs from frozen campaign")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print("native_waveform_replayed", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        npk = fitted_peak(nt, nm, "quadratic_power")
        rpk = fitted_peak(rt, rm, "quadratic_power")
        na = frequency_anchor(nt, nm, mass_seconds, 0.1, npk["time_s"])
        ra = frequency_anchor(rt, rm, mass_seconds, 0.1, rpk["time_s"])
        if abs(ra["time_s"] - baseline["frequency_anchored"]["reference_anchor"]["time_s"]) > 1e-12:
            raise ValueError("pinned public frequency anchor changed")
        nx, nh = _aligned_series(nt, nm, mass_seconds, na, ra["peak_lag_M"])
        rx, rh = _aligned_series(rt, rm, mass_seconds, ra, ra["peak_lag_M"])
        report["samples"] = []
        for t in plan["relative_times_M"]:
            n, r = _interp_complex(nx, nh, t), _interp_complex(rx, rh, t)
            report["samples"].append(dict(
                time_from_reference_peak_M=t,
                native_over_reference_amplitude=float(abs(n)/abs(r)),
                native_minus_reference_phase_rad=float(np.angle(n*r.conjugate()))))
        report["slopes"] = [dict(window_M=[lo, hi], native=_slope(nx, nh, lo, hi),
                                  reference=_slope(rx, rh, lo, hi))
                             for lo, hi in plan["slope_windows_M"]]
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
